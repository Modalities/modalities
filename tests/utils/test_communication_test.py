import pytest
import torch
import torch.multiprocessing as mp

from modalities.config.config import ProcessGroupBackendType
from modalities.utils.communication_test import run_communication_test
from tests.end2end_tests.custom_components import MultiProcessingCudaEnv


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="This test requires 2 GPUs.",
)
class TestCommunicationTest:
    @staticmethod
    def test_run_communication_test():
        """
        Test to ensure that the communication test runs without errors.
        This is a simple smoke test to verify that the communication setup is functional.
        """
        world_size = 2
        mp.spawn(
            TestCommunicationTest._test_run_communication_test_wrapper,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )

    @staticmethod
    def _test_run_communication_test_wrapper(
        process_id: int,
        world_size: int,
    ):
        """This function is a wrapper for the communication test to run it in a multiprocessing setup."""
        with MultiProcessingCudaEnv(
            process_group_backend=ProcessGroupBackendType.nccl,
            global_rank=process_id,
            local_rank=process_id,
            world_size=world_size,
            rdvz_port=22356,
        ):
            TestCommunicationTest._test_run_communication_test_impl()

    @staticmethod
    def _test_run_communication_test_impl():
        """
        Implementation of the communication test.
        This function is called by the multiprocessing wrapper.
        """
        run_communication_test()
