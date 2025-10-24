import tempfile
from multiprocessing import Queue

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from modalities.training.gradient_clipping.fsdp_gradient_clipper import FSDP2GradientClipper
from tests.utility import find_free_port


def test_pipeline_parallelized_clipping_equivalent_to_single_stage_clipping():
    max_norm = 0.1
    # create full model and initialize deterministically
    torch.manual_seed(42)
    full = FullModel()

    # create an input and compute gradients on the full model
    x = torch.randn(2, 4)
    out = full(x)
    loss = out.pow(2).sum()
    loss.backward()

    # save full model state and grads to a temporary file for workers
    state = {}
    for name, p in full.named_parameters():
        # store parameter data and grads on CPU
        state[name] = p.data.cpu().clone()
    grads = {}
    for name, p in full.named_parameters():
        grads[name] = p.grad.cpu().clone()

    with tempfile.NamedTemporaryFile() as tmp:
        store_path = tmp.name
        torch.save({"state": state, "grads": grads}, store_path)

        # set up multiprocessing to simulate 2 pipeline stages
        world_size = 2
        port = find_free_port()
        q = mp.get_context("spawn").Queue()
        mp.spawn(_worker, args=(world_size, store_path, port, max_norm, q), nprocs=world_size, join=True)

        # collect results
        results = {}
        for _ in range(world_size):
            rank, coll = q.get()
            results[rank] = coll

        # perform clipping on the full model (single-stage)
        FSDP2GradientClipper.clip_grad_norm_(
            parameters=full.parameters(),
            max_norm=max_norm,
            norm_type=2.0,
            error_if_nonfinite=True,
            foreach=True,
            device_mesh=None,
        )

        # compare full model parts to the per-stage results
        full_a_params = [p.data.cpu() for p in full.a.parameters()]
        full_b_params = [p.data.cpu() for p in full.b.parameters()]

        # ranks: 0 -> partA, 1 -> partB
        assert 0 in results and 1 in results

        for p_full, p_pp in zip(full_a_params, results[0]):
            t_pp = torch.as_tensor(p_pp, dtype=p_full.dtype)
            assert torch.allclose(p_full, t_pp, atol=1e-6, rtol=1e-5)

        for p_full, p_pp in zip(full_b_params, results[1]):
            t_pp = torch.as_tensor(p_pp, dtype=p_full.dtype)
            assert torch.allclose(p_full, t_pp, atol=1e-6, rtol=1e-5)


class PartA(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 5, bias=False)

    def forward(self, x: torch.Tensor):
        return self.lin(x)


class PartB(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(5, 3, bias=False)

    def forward(self, x: torch.Tensor):
        return self.lin(x)


class FullModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = PartA()
        self.b = PartB()

    def forward(self, x: torch.Tensor):
        return self.b(self.a(x))


def _worker(rank: int, world_size: int, store_path: str, port: int, max_norm: float, q: Queue):
    # initialize distributed
    dist.init_process_group(backend="gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world_size)

    # load saved full model state and grads
    data = torch.load(store_path)
    state = data["state"]
    grads = data["grads"]

    # create the corresponding part for this rank and load weights
    if rank == 0:
        part = PartA()
        # map parameters from full model: a.lin.weight
        part.lin.weight.data.copy_(state["a.lin.weight"])
        # assign gradients
        for name, p in part.named_parameters():
            full_name = f"a.{name}"
            if full_name in grads:
                p.grad = grads[full_name].clone()
    else:
        part = PartB()
        part.lin.weight.data.copy_(state["b.lin.weight"])
        for name, p in part.named_parameters():
            full_name = f"b.{name}"
            if full_name in grads:
                p.grad = grads[full_name].clone()

    # create a dummy device_mesh-like object that matches the parts of DeviceMesh
    # expected by get_mesh_for_parallelism_method and FSDP2GradientClipper.
    class DummyPPMesh:
        def __init__(self, group):
            self._group = group

        def get_group(self):
            return self._group

    class DummyDeviceMesh:
        def __init__(self, group):
            # include the PP mesh name so get_mesh_for_parallelism_method finds it
            self.mesh_dim_names = ("pp",)
            self._pp = DummyPPMesh(group)

        def __getitem__(self, name: str):
            if name == "pp":
                return self._pp
            raise KeyError(name)

    mesh = DummyDeviceMesh(dist.group.WORLD)

    # call the clipping function which will perform all_reduce across the pp group
    FSDP2GradientClipper.clip_grad_norm_(
        parameters=part.parameters(),
        max_norm=max_norm,
        norm_type=2.0,
        error_if_nonfinite=True,
        foreach=True,
        device_mesh=mesh,
    )

    # collect clipped parameter tensors (cpu) and serialize to plain Python lists
    # to avoid multiprocessing shared-storage pickling issues.
    collected = [p.data.cpu().numpy().tolist() for p in part.parameters()]
    q.put((rank, collected))

    dist.destroy_process_group()
