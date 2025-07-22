import os
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

from modalities.__main__ import Main
from modalities.batch import EvaluationResultBatch
from modalities.config.config import ProcessGroupBackendType
from modalities.config.instantiation_models import TrainingComponentsInstantiationModel
from modalities.logging_broker.messages import Message
from tests.end2end_tests.custom_components import (
    MultiProcessingCudaEnv,
    SaveAllResultSubscriber,
    SaveAllResultSubscriberConfig,
)


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="This e2e test requires 2 GPUs.",
)
class TestFSDPLossConvergence:
    @staticmethod
    def _test_fsdp_loss_convergence_thread(process_id: int, world_size: int, rdvz_port: int, config_file_path: Path):
        # Important: we fix the seed to make sure that the mnodel weights are the same across all ranks
        torch.manual_seed(20)
        torch.cuda.manual_seed(20)

        components = TestFSDPLossConvergence._run_modalities_training(
            process_id=process_id, world_size=world_size, rdvz_port=rdvz_port, config_file_path=config_file_path
        )
        # collect results to assert that the loss has gone down
        result_messages: list[Message[EvaluationResultBatch]] = components.evaluation_subscriber.message_list
        assert (
            result_messages[0].payload.losses["train loss avg"].value
            > result_messages[-1].payload.losses["train loss avg"].value
        )
        for i, message in enumerate(result_messages):
            print(f"step {i}: {message.payload.losses['train loss avg'].value}")

    @staticmethod
    def _run_modalities_training(
        process_id: int, world_size: int, rdvz_port: int, config_file_path: Path
    ) -> TrainingComponentsInstantiationModel:
        with MultiProcessingCudaEnv(
            process_group_backend=ProcessGroupBackendType.nccl,
            global_rank=process_id,
            local_rank=process_id,
            world_size=world_size,
            rdvz_port=rdvz_port,
        ):
            main_obj = Main(config_file_path)
            # register custom results subscriber for tracking all results
            main_obj.add_custom_component(
                component_key="results_subscriber",
                variant_key="save_all",
                custom_component=SaveAllResultSubscriber,
                custom_config=SaveAllResultSubscriberConfig,
            )
            # build the components (indluduing the custom component)
            components: TrainingComponentsInstantiationModel = main_obj.build_components(
                components_model_type=TrainingComponentsInstantiationModel
            )
            # run the training run
            main_obj.run(components)
        return components

    @staticmethod
    @pytest.mark.parametrize(
        "rdvz_port, relative_config_path",
        [
            (22801, "configs/fsdp1_gpt2_train_num_steps_8.yaml"),
            (22802, "configs/fsdp2_gpt2_train_num_steps_8.yaml"),
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
