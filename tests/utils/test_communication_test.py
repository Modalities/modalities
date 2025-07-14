import os

import pytest
import torch

from modalities.config.config import ProcessGroupBackendType
from modalities.running_env.cuda_env import CudaEnv
from modalities.utils.communication_test import run_communication_test


@pytest.mark.skipif(
    "RANK" not in os.environ or torch.cuda.device_count() < 2,
    reason="This test requires 2 GPUs and a torchrun distributed environment.",
)
def test_run_communication_test():
    """
    Test to ensure that the communication test runs without errors.
    This is a simple smoke test to verify that the communication setup is functional.
    """
    with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
        run_communication_test()
