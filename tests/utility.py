import os
import socket
import time
from multiprocessing import Queue
from multiprocessing.managers import SyncManager

import debugpy
import torch
import torch.distributed as dist
from torch.distributed._tensor import DeviceMesh
from torch.multiprocessing.spawn import ProcessContext

from modalities.batch import DatasetBatch


def find_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def add_debugger_to_distributed_test():
    """Add a debugger to a distributed test.
    This function should be called at the beginning of the test.

    Within VScode you can use the following configuration to attach the debugger to the test:

    ```json
    {
        "name": "Test Torch Distributed",
        "type": "python",
        "request": "launch",
        "program": "path/to/torchrun",
        "console": "integratedTerminal",
        "env": {"CUDA_VISIBLE_DEVICES": "0,1"},
        "args": ["--rdzv-endpoint", "localhost:29833", "--nnodes", "1",
                "--nproc_per_node", "2", "path/to/pytest", "tests/some_test.py"],
        "justMyCode": false,
    },
    ```
    """
    # Get the rank of the process (0 or 1 in this case)
    rank = int(os.getenv("RANK"))

    # Use a different port for each process
    port = 9875 + rank
    debugpy.listen(("0.0.0.0", port))  # Listening on all interfaces to allow debugger to attach
    print(f"Rank {rank}: Waiting for debugger to attach on port {port}...")
    debugpy.wait_for_client()  # Pause here until the debugger attaches


def configure_dataloader_mock(
    batch_size: int,
    seq_len: int,
    num_batches: int,
    sample_key: str,
    target_key: str,
    llm_data_loader_mock,
):
    sample_tensor = torch.randint(size=(batch_size, seq_len), low=1, high=100)
    samples = {sample_key: sample_tensor[:, :-1]}
    targets = {target_key: sample_tensor[:, 1:]}

    batches = [DatasetBatch(targets=targets, samples=samples) for _ in range(num_batches)]

    llm_data_loader_mock.__iter__ = lambda _: iter(batches)
    llm_data_loader_mock.batch_size = batch_size
    llm_data_loader_mock.fast_forward_batch_id = 0
    llm_data_loader_mock.__len__ = lambda _: num_batches

    return llm_data_loader_mock, batches


def monitor_child_processes(
    manager: SyncManager,
    error_queue: Queue,
    proc_ctx: ProcessContext,
) -> None:
    # Normalize the return value from mp.spawn. When join=False it often
    # returns a ProcessContext-like object that may expose a `processes`
    # attribute. Other implementations may return an iterable of Process
    # objects. Build a `processes` list defensively so we can monitor and
    # terminate child processes below without assuming a particular type.
    processes = []
    if proc_ctx is None:
        processes = []
    else:
        # common attribute names that might hold the list of processes
        candidate_attrs = ["processes", "_processes", "workers", "process_list", "processes_"]
        found = False
        for attr in candidate_attrs:
            if hasattr(proc_ctx, attr):
                ps = getattr(proc_ctx, attr)
                try:
                    processes = list(ps)
                except Exception:
                    processes = [ps]
                found = True
                break
        if not found:
            # If proc_ctx itself is iterable, exhaust it into a list
            try:
                processes = list(proc_ctx)
            except Exception:
                # Fallback: if proc_ctx behaves like a single process-like
                # object (has terminate/is_alive/join), wrap it in a list.
                if hasattr(proc_ctx, "terminate") or hasattr(proc_ctx, "is_alive") or hasattr(proc_ctx, "join"):
                    processes = [proc_ctx]
                else:
                    processes = []

    # Monitor the error queue and child processes. If any child reports an exception,
    # terminate the other workers and raise the error in the parent to fail the test fast.
    try:
        # Loop until all processes finished or an error is reported
        while True:
            # If an error was reported by any child process, terminate remaining children
            if not error_queue.empty():
                proc_id, tb = error_queue.get()
                # terminate and join all processes (or the proc_ctx wrapper)
                for p in processes:
                    try:
                        if hasattr(p, "is_alive"):
                            alive = p.is_alive()
                        elif hasattr(p, "exitcode"):
                            alive = getattr(p, "exitcode") is None
                        else:
                            alive = True
                        if alive and hasattr(p, "terminate"):
                            p.terminate()
                    except Exception:
                        pass
                # If we didn't find individual process objects but proc_ctx
                # exposes a terminate method, call it as a fallback.
                try:
                    if not processes and hasattr(proc_ctx, "terminate"):
                        proc_ctx.terminate()
                except Exception:
                    pass

                for p in processes:
                    try:
                        if hasattr(p, "join"):
                            p.join(timeout=5)
                    except Exception:
                        pass
                try:
                    if hasattr(proc_ctx, "join"):
                        proc_ctx.join(timeout=1)
                except Exception:
                    pass
                raise AssertionError(f"Child process {proc_id} raised an exception:\n{tb}")

            # If all processes have finished, break
            all_finished = all((not p.is_alive()) for p in processes)
            if all_finished:
                # join them to collect exitcodes
                for p in processes:
                    try:
                        p.join()
                    except Exception:
                        pass
                # If we have a ProcessContext, call its join to clean up as well
                try:
                    if hasattr(proc_ctx, "join"):
                        proc_ctx.join(timeout=1)
                except Exception:
                    pass
                break

            time.sleep(0.05)
    finally:
        try:
            manager.shutdown()
        except Exception:
            pass


def tensors_equal_across_mesh(tensor: torch.Tensor, device_mesh: DeviceMesh) -> bool:
    """Check if tensors are equal across all ranks in the mesh"""
    process_group = device_mesh.get_group()
    device = tensor.device

    gathered_tensors = [torch.zeros_like(tensor, device=device) for _ in range(process_group.size())]
    dist.all_gather(gathered_tensors, tensor, group=process_group)

    reference = gathered_tensors[0]
    return all(torch.equal(reference, t) for t in gathered_tensors)


def tensors_pairwise_not_equal_across_mesh(tensor: torch.Tensor, device_mesh: DeviceMesh) -> bool:
    """Check if tensors are pairwise not equal across all ranks in the mesh"""
    process_group = device_mesh.get_group()
    device = tensor.device

    gathered_tensors = [torch.zeros_like(tensor, device=device) for _ in range(process_group.size())]
    dist.all_gather(gathered_tensors, tensor, group=process_group)

    for i in range(len(gathered_tensors)):
        for j in range(i + 1, len(gathered_tensors)):
            if torch.equal(gathered_tensors[i], gathered_tensors[j]):
                return False
    return True
