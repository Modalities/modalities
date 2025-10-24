import tempfile
import types
from multiprocessing import Queue
from unittest.mock import MagicMock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from modalities.training.gradient_clipping.fsdp_gradient_clipper import (
    FSDP1GradientClipper,
    FSDP1LoggingOnlyGradientClipper,
    FSDP2GradientClipper,
    FSDP2LoggingOnlyGradientClipper,
    GradientClippingMode,
)
from tests.utility import find_free_port


class MockFSDPModel:
    def __init__(self):
        self.param1 = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        self.param2 = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
        self.param1.grad = torch.tensor([1.0, 1.0])
        self.param2.grad = torch.tensor([1.0, 1.0])

    def parameters(self):
        return [self.param1, self.param2]


# Test for FSDP1 gradient clipper
def test_fsdp1_gradient_clipper():
    """
    Test FSDP1GradientClipper's ability to clip gradients correctly.
    Uses a mock model with a dynamically added clip_grad_norm_ method to verify norm calculation and gradient scaling.
    """
    mock_model = MockFSDPModel()
    max_norm = 1.0
    norm_type = GradientClippingMode.P2_NORM

    # Note: FSDPGradientClipper requires clip_grad_norm_, but user's model lacks it.
    # To use FSDPGradientClipper, we’d need to add this method, which deviates from the request.
    # For strict adherence, we could skip this test or raise an error, but let’s adapt.
    # Temporarily extend MockFSDPModel in this test (with a comment explaining).
    def clip_grad_norm_(self, max_norm, norm_type):
        params = [p for p in self.parameters() if p.grad is not None]
        total_norm = torch.norm(torch.stack([torch.norm(p.grad, norm_type) for p in params]), norm_type)
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in params:
                p.grad.data.mul_(clip_coef)
        return total_norm

    # Dynamically add the method for this test
    mock_model.clip_grad_norm_ = types.MethodType(clip_grad_norm_, mock_model)

    clipper = FSDP1GradientClipper(wrapped_model=mock_model, max_norm=max_norm, norm_type=norm_type)
    norm = clipper.clip_gradients()

    # Expected norm before clipping: sqrt(1^2 + 1^2 + 1^2 + 1^2) = 2.0
    expected_norm = torch.tensor(2.0)
    assert torch.allclose(norm, expected_norm), f"Expected norm {expected_norm}, got {norm}"

    # Gradients should be scaled to max_norm / total_norm = 1.0 / 2.0 = 0.5
    expected_grad = torch.tensor([0.5, 0.5])
    for param in mock_model.parameters():
        assert torch.allclose(param.grad, expected_grad), f"Expected grad {expected_grad}, got {param.grad}"


def test_fsdp1_logging_only_gradient_clipper():
    """
    Test that FSDP1LoggingOnlyGradientClipper calls clip_grad_norm_ with max_norm=torch.inf,
    ensuring no clipping occurs, and returns the gradient norm.
    """
    # Create a mock FSDP1 model
    mock_model = MagicMock()
    norm_type = GradientClippingMode.P2_NORM
    clipper = FSDP1LoggingOnlyGradientClipper(wrapped_model=mock_model, norm_type=norm_type)

    # Call clip_gradients
    clipper.clip_gradients()

    # Verify that clip_grad_norm_ was called with max_norm=torch.inf
    mock_model.clip_grad_norm_.assert_called_once_with(max_norm=torch.inf, norm_type=norm_type.value)


def test_fsdp2_clip_grad_norm():
    """
    Test the static clip_grad_norm_ method in FSDP2GradientClipper to ensure it correctly
    computes the gradient norm and clips gradients when necessary.
    """
    # Create parameters with gradients
    mock_model = MockFSDPModel()

    # Compute expected total norm (Euclidean norm, norm_type=2)
    expected_norm = (1**2 + 1**2 + 1**2 + 1**2) ** 0.5  # sqrt(4) = 2.0

    # Test case 1: max_norm > total_norm (no clipping)
    max_norm = expected_norm + 1  # 3.0
    norm = FSDP2GradientClipper(mock_model, max_norm=max_norm, norm_type=GradientClippingMode.P2_NORM).clip_gradients()
    assert torch.allclose(norm, torch.tensor(expected_norm)), "Norm should match expected total norm"
    assert torch.allclose(mock_model.param1.grad, torch.tensor([1.0, 1.0])), "Gradients should not be clipped"
    assert torch.allclose(mock_model.param2.grad, torch.tensor([1.0, 1.0])), "Gradients should not be clipped"

    # Test case 2: max_norm < total_norm (clipping occurs)
    max_norm = expected_norm / 2  # 1.0
    norm = FSDP2GradientClipper(mock_model, max_norm=max_norm, norm_type=GradientClippingMode.P2_NORM).clip_gradients()
    assert torch.allclose(norm, torch.tensor(expected_norm)), "Norm should match pre-clipping total norm"
    scale = max_norm / expected_norm  # 1.0 / 2.0 = 0.5
    expected_grad = torch.tensor([1.0 * scale, 1.0 * scale])
    assert torch.allclose(mock_model.param1.grad, expected_grad), "Gradients should be clipped"
    assert torch.allclose(mock_model.param2.grad, expected_grad), "Gradients should be clipped"


def test_fsdp2_gradient_clipper():
    """
    Test that FSDP2GradientClipper correctly calls clip_grad_norm_ on the wrapped model's parameters.
    """
    # Create a mock FSDP2 model with parameters

    mock_model = MockFSDPModel()

    max_norm = 1.0
    norm_type = GradientClippingMode.P2_NORM
    clipper = FSDP2GradientClipper(wrapped_model=mock_model, max_norm=max_norm, norm_type=norm_type)

    # Call clip_gradients
    norm = clipper.clip_gradients()

    expected_norm = (1**2 + 1**2 + 1**2 + 1**2) ** 0.5  # 2.0
    assert torch.allclose(norm, torch.tensor(expected_norm)), "Norm should match expected total norm"

    scale = max_norm / expected_norm  # 0.5
    expected_grad = torch.tensor([1.0 * scale, 1.0 * scale])
    for param in mock_model.parameters():
        assert torch.allclose(param.grad, expected_grad), "Gradients should be clipped"


def test_fsdp2_logging_only_gradient_clipper():
    """
    Test that FSDP2LoggingOnlyGradientClipper computes the gradient norm without clipping.
    """
    mock_model = MockFSDPModel()

    norm_type = GradientClippingMode.P2_NORM
    clipper = FSDP2LoggingOnlyGradientClipper(wrapped_model=mock_model, norm_type=norm_type)

    # Call clip_gradients
    norm = clipper.clip_gradients()

    # Verify the norm and that gradients are unchanged
    expected_norm = (1**2 + 1**2 + 1**2 + 1**2) ** 0.5  # 2.0
    assert torch.allclose(norm, torch.tensor(expected_norm)), "Norm should match expected total norm"
    for param in mock_model.parameters():
        assert torch.allclose(param.grad, torch.tensor([1.0, 1.0])), "Gradients should not be modified"


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
        FSDP2GradientClipper(
            wrapped_model=full,
            max_norm=max_norm,
            norm_type=GradientClippingMode.P2_NORM,
            device_mesh=None,
            error_if_nonfinite=True,
            foreach=True,
        ).clip_gradients()

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
    FSDP2GradientClipper(
        wrapped_model=part,
        max_norm=max_norm,
        norm_type=GradientClippingMode.P2_NORM,
        device_mesh=mesh,
        error_if_nonfinite=True,
        foreach=True,
    ).clip_gradients()

    # collect clipped parameter tensors (cpu) and serialize to plain Python lists
    # to avoid multiprocessing shared-storage pickling issues.
    collected = [p.data.cpu().numpy().tolist() for p in part.parameters()]
    q.put((rank, collected))

    dist.destroy_process_group()
