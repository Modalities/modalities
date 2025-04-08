import os
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

from modalities.batch import EvaluationResultBatch
from modalities.logging_broker.messages import Message
from tests.end2end_tests.system_tests.run_modalities_entrypoints import run_modalities_training


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="This e2e test requires 2 GPUs.",
)
class TestFSDPLossConvergence:
    @staticmethod
    def _test_fsdp_loss_convergence_thread(process_id: int, world_size: int, rdvz_port: int, config_file_path: Path):
        components = run_modalities_training(
            process_id=process_id, world_size=world_size, rdvz_port=rdvz_port, config_file_path=config_file_path
        )
        # collect results to assert that the loss has gone down
        result_messages: list[Message[EvaluationResultBatch]] = components.evaluation_subscriber.message_list
        assert (
            result_messages[0].payload.losses["train loss avg"].value
            > result_messages[-1].payload.losses["train loss avg"].value
        )

    @staticmethod
    @pytest.mark.parametrize(
        "rdvz_port, relative_config_path",
        [
            (22355, "configs/fsdp1_gpt2_train_num_steps_8.yaml"),
            (22356, "configs/fsdp2_gpt2_train_num_steps_8.yaml"),
        ],
    )
    def test_fsdp_loss_convergence(rdvz_port, relative_config_path: str):
        working_dir = Path(os.path.dirname(__file__))
        config_file_path = working_dir / relative_config_path
        world_size = 2
        mp.spawn(
            TestFSDPLossConvergence._test_fsdp_loss_convergence_thread,
            args=(world_size, rdvz_port, config_file_path),
            nprocs=world_size,
            join=True,
        )
