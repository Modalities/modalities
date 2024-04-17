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
        with tempfile.TemporaryDirectory() as temp_dir:
            # config for two steps model
            gpt2_two_steps_config_file_path = Path("tests/end2end_tests/gpt2_two_steps.yaml")
            gpt2_two_steps_config_dict = load_app_config_dict(gpt2_two_steps_config_file_path)
            
            # adopt the checkpoint path
            gpt2_two_steps_config_dict["checkpointing"]["config"]["checkpointing_execution"]["config"]["checkpoint_path"]= str(Path(temp_dir))
            gpt2_two_steps_config_dict["settings"]["experiment_id"] = "0"

            # config for one step model
            gpt2_warm_start_from_step_1_config_file_path = Path("tests/end2end_tests/gpt2_warm_start_from_step_1.yaml")
            gpt2_warm_start_from_step_1_dict = load_app_config_dict(gpt2_two_steps_config_file_path)

            # adopt the checkpoint path
            gpt2_warm_start_from_step_1_dict["checkpointing"]["config"]["checkpointing_execution"]["config"]["checkpoint_path"]= str(Path(temp_dir)) + "/0/"
            gpt2_two_steps_config_dict["wrapped_model"]["config"]["checkpoint_path"] = str(Path(temp_dir)) + "/0/"


            main_obj_1 = Main(gpt2_two_steps_config_dict, gpt2_two_steps_config_file_path)
            with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
                main_obj_1.add_custom_component(
                    component_key="results_subscriber",
                    variant_key="save_all",
                    custom_component=SaveAllResultSubscriber,
                    custom_config=SaveAllResultSubscriberConfig,
                )
                components_1 = main_obj_1.build_components(components_model_type=TrainingComponentsInstantiationModel)
                main_obj_1.run(components_1)
            
            main_obj_2 = Main(gpt2_warm_start_from_step_1_dict, gpt2_warm_start_from_step_1_config_file_path)
            with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
                main_obj_2.add_custom_component(
                    component_key="results_subscriber",
                    variant_key="save_all",
                    custom_component=SaveAllResultSubscriber,
                    custom_config=SaveAllResultSubscriberConfig,
                )
                components_2 = main_obj_2.build_components(components_model_type=TrainingComponentsInstantiationModel)
                main_obj_2.run(components_2)
            
            messages_1: List[Message[EvaluationResultBatch]] = components_1.evaluation_subscriber.message_list
            messages_2: List[Message[EvaluationResultBatch]] = components_2.evaluation_subscriber.message_list

            key="CLMCrossEntropyLoss interval average"
            assert messages_1[-1].payload.losses[key] == messages_2[-1].payload.losses[key] 
            # messages_1: List[Message[EvaluationResultBatch]] = components_1.evaluation_subscriber.message_list
            # expected_values = {'CLMCrossEntropyLoss interval average': 10.7891}
            # for key, expected in expected_values.items():
            #     assert key in messages_1[-1].payload.losses, f"Key '{key}' not found in observed values"
            #     observed = messages_1[-1].payload.losses[key]
            #     assert observed == expected, f"AssertionError: {observed} != {expected} for key '{key}'"
