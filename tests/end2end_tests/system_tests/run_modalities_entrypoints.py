import os
from pathlib import Path

from modalities.__main__ import Main
from modalities.config.config import ProcessGroupBackendType
from modalities.config.instantiation_models import TrainingComponentsInstantiationModel
from modalities.running_env.cuda_env import CudaEnv
from tests.end2end_tests.custom_components import SaveAllResultSubscriber, SaveAllResultSubscriberConfig


def run_modalities_training(process_id: int, world_size: int, rdvz_port: int, config_file_path: Path):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(rdvz_port)
    os.environ["RANK"] = str(process_id)
    os.environ["LOCAL_RANK"] = str(process_id)
    os.environ["WORLD_SIZE"] = str(world_size)

    with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
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
