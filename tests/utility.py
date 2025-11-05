import os
import socket

import debugpy
import torch

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
