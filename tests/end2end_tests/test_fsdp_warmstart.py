import tempfile
from pathlib import Path
from typing import List

from pydantic import BaseModel

from modalities.__main__ import Main, load_app_config_dict
from modalities.batch import EvaluationResultBatch
from modalities.config.config import ProcessGroupBackendType, TrainingComponentsInstantiationModel
from modalities.logging_broker.messages import Message
from modalities.logging_broker.subscriber import MessageSubscriberIF
from modalities.running_env.cuda_env import CudaEnv

# NOTE: We need to run the tests in a torch distributed environment with at least two GPUs.
# CUDA_VISIBLE_DEVICES=0,1 torchrun --rdzv-endpoint localhost:29502 --nnodes 1 --nproc_per_node 2 \
#   $(which pytest) path/to/test_fsdp_to_disc_checkpointing.py


_ROOT_DIR = Path(__file__).parents[1]


class SaveAllResultSubscriber(MessageSubscriberIF[EvaluationResultBatch]):
    def __init__(self):
        self.message_list: List[Message[EvaluationResultBatch]] = []

    def consume_message(self, message: Message[EvaluationResultBatch]):
        """Consumes a message from a message broker."""
        self.message_list.append(message)


class SaveAllResultSubscriberConfig(BaseModel):
    pass


# @pytest.mark.skipif(
#     "RANK" not in os.environ or torch.cuda.device_count() < 2,
#     reason="This e2e test requires 2 GPUs and a torchrun distributed environment.",
# )
class TestWarmstart:
    def test_warm_start(self):
        with tempfile.TemporaryDirectory():
            gpt2_two_steps_config_file_path = Path("tests/end2end_tests/gpt2_two_steps.yaml")
            gpt2_two_steps_config_dict = load_app_config_dict(gpt2_two_steps_config_file_path)
            # gpt2_two_steps_config_dict["checkpointing"]["config"]["checkpointing_execution"]
            # ["config"]["checkpoint_path"]
            # = temporary_checkpoint_folder_path
            # gpt2_two_steps_config_dict["settings"]["experiment_id"] = "0"

            main_obj = Main(gpt2_two_steps_config_dict, gpt2_two_steps_config_file_path)
            with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
                main_obj.add_custom_component(
                    component_key="results_subscriber",
                    variant_key="save_all",
                    custom_component=SaveAllResultSubscriber,
                    custom_config=SaveAllResultSubscriberConfig,
                )
                components_1 = main_obj.build_components(components_model_type=TrainingComponentsInstantiationModel)
                main_obj.run(components_1)
            messages_1: List[Message[EvaluationResultBatch]] = components_1.evaluation_subscriber.message_list
            assert messages_1[-1].payload.losses is None
