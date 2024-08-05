import os

from modalities.config.component_factory import ComponentFactory
from modalities.config.config import ProcessGroupBackendType, load_app_config_dict
from modalities.config.instantiation_models import TrainingComponentsInstantiationModel
from modalities.registry.components import COMPONENTS
from modalities.registry.registry import Registry
from modalities.running_env.cuda_env import CudaEnv


def setup(rank, world_size):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"


setup(0, 1)


def extract_model(config_path):
    config_dict = load_app_config_dict(config_path)
    registry = Registry(COMPONENTS)

    component_factory = ComponentFactory(registry=registry)
    with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
        components = component_factory.build_components(
            config_dict=config_dict,
            components_model_type=TrainingComponentsInstantiationModel,
        )
    print(components)
    return components
