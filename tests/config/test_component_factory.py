from pathlib import Path
from typing import Dict, Union

import pytest
from omegaconf import OmegaConf

from modalities.config.component_factory import ComponentFactory
from tests.config.components import ComponentV, ComponentW, ComponentY
from tests.config.configs import CompVConfig, CompWConfig, CompXConfig, CompYConfig, CustomComp1Config, ReferenceConfig


def load_app_config_dict(config_file_path: Path) -> Dict:
    cfg = OmegaConf.load(config_file_path)
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    return config_dict


@pytest.mark.parametrize(
    "config_file_path",
    [
        Path("tests/config/test_configs/config_backward_reference.yaml"),
        Path("tests/config/test_configs/config_forward_reference.yaml"),
    ],
)
def test_backward_reference(config_file_path: Path):
    comp_config_types = Union[CompXConfig, CompYConfig, CompWConfig, CompVConfig, ReferenceConfig]
    component_names = ["comp_x_1", "comp_y_1"]

    config_dict = load_app_config_dict(config_file_path=config_file_path)

    components = ComponentFactory.build_config(
        config_dict=config_dict, config_types=comp_config_types, component_names=component_names
    )

    # make sure that the reference is not identical, despite both being of type COMP_W
    assert components["comp_x_1"].single_dependency != components["comp_y_1"].multi_dependency[0]
    # make sure that the reference is identical, since we are referencing comp_x_1 in the multi depencency of comp_y_1
    assert components["comp_x_1"] == components["comp_y_1"].multi_dependency[2]


@pytest.mark.parametrize(
    "config_file_path",
    [
        Path("tests/config/test_configs/config_non_existing_reference.yaml"),
    ],
)
def test_non_existing_reference(config_file_path: Path):
    comp_config_types = Union[CompXConfig, CompYConfig, CompWConfig, CompVConfig, ReferenceConfig]
    component_names = ["comp_x_1", "comp_y_1"]

    config_dict = load_app_config_dict(config_file_path=config_file_path)

    with pytest.raises(KeyError):
        ComponentFactory.build_config(
            config_dict=config_dict, config_types=comp_config_types, component_names=component_names
        )


@pytest.mark.parametrize(
    "config_file_path",
    [
        Path("tests/config/test_configs/config_hierarchical_list_component.yaml"),
    ],
)
def test_hierarchical_component_instantiation(config_file_path: Path):
    comp_config_types = Union[CompYConfig, CompWConfig, CompVConfig, ReferenceConfig]
    component_names = ["comp_y_1"]

    config_dict = load_app_config_dict(config_file_path=config_file_path)

    components = ComponentFactory.build_config(
        config_dict=config_dict, config_types=comp_config_types, component_names=component_names
    )

    assert isinstance(components["comp_y_1"].multi_dependency[0], ComponentW)
    assert isinstance(components["comp_y_1"].multi_dependency[1], ComponentV)
    assert isinstance(components["comp_y_1"], ComponentY)


@pytest.mark.parametrize(
    "config_file_path",
    [
        Path("tests/config/test_configs/config_hierarchical_list_component.yaml"),
    ],
)
def test_component_filter(config_file_path: Path):
    comp_config_types = Union[CompYConfig, CompWConfig, CompVConfig, ReferenceConfig]
    component_names = ["comp_y_1"]

    config_dict = load_app_config_dict(config_file_path=config_file_path)

    components = ComponentFactory.build_config(
        config_dict=config_dict, config_types=comp_config_types, component_names=component_names
    )
    assert "comp_y_1" in components

    component_names += "abc"
    with pytest.raises(KeyError):
        components = ComponentFactory.build_config(
            config_dict=config_dict, config_types=comp_config_types, component_names=component_names
        )


@pytest.mark.parametrize(
    "config_file_path",
    [
        Path("tests/config/test_configs/config_custom_component.yaml"),
    ],
)
def test_custom_component(config_file_path: Path):
    comp_config_types = Union[CompYConfig, CompWConfig, CompVConfig, ReferenceConfig]
    comp_config_types = comp_config_types | CustomComp1Config
    component_names = ["custom_comp_1"]

    config_dict = load_app_config_dict(config_file_path=config_file_path)

    components = ComponentFactory.build_config(
        config_dict=config_dict, config_types=comp_config_types, component_names=component_names
    )
    assert "custom_comp_1" in components
