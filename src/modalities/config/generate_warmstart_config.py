from pathlib import Path

from modalities.config.component_factory import ComponentFactory
from modalities.config.config import load_app_config_dict
from modalities.config.instantiation_models import TrainingComponentsInstantiationModel
from modalities.registry.components import COMPONENTS
from modalities.registry.registry import Registry


def get_warmstart_config(prev_config_path: Path, warmstart_config_path: Path):
    prev_config_dict = load_app_config_dict(prev_config_path)
    # warmstart_config_dict = load_app_config_dict(warmstart_config_path)
    registry = Registry(COMPONENTS)
    component_factory = ComponentFactory(registry=registry)

    settings = component_factory.build_components(
        config_dict=prev_config_dict["settings"], components_model_type=TrainingComponentsInstantiationModel.Settings
    )
    print(settings)


if __name__ == "__main__":
    prev_config_path = Path(
        "/raid/s3/opengptx/max_lue/repositories/modalities/tests/end2end_tests/gpt2_train_num_steps_8.yaml"
    )
    warmstart_config_path = Path("tmp_config.yaml")
    get_warmstart_config(prev_config_path, warmstart_config_path)
