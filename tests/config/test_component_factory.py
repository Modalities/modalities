from pathlib import Path

import pytest

from modalities.__main__ import load_app_config_dict
from modalities.config.component_factory import ComponentFactory
from modalities.registry.registry import Registry
from tests.config.components import ComponentV, ComponentW, ComponentX, ComponentY
from tests.config.configs import CompVConfig, CompWConfig, CompXConfig, CompYConfig


@pytest.fixture
def component_factory() -> ComponentFactory:
    components = [
        ("COMP_V", "default", ComponentV),
        ("COMP_W", "default", ComponentW),
        ("COMP_Y", "default", ComponentY),
        ("COMP_X", "default", ComponentX),
    ]

    component_registry = Registry()
    for component in components:
        component_registry.add_entity(*component)

    component_config_registry = Registry()
    configs = [
        ("COMP_V", "default", CompVConfig),
        ("COMP_W", "default", CompWConfig),
        ("COMP_X", "default", CompXConfig),
        ("COMP_Y", "default", CompYConfig),
    ]
    for config in configs:
        component_config_registry.add_entity(*config)

    factory = ComponentFactory(config_registry=component_config_registry, component_registry=component_registry)
    return factory


@pytest.mark.parametrize(
    "config_file_path",
    [
        Path("tests/config/test_configs/config_backward_reference.yaml"),
        Path("tests/config/test_configs/config_forward_reference.yaml"),
    ],
)
def test_backward_reference(config_file_path: Path, component_factory: ComponentFactory):
    component_names = ["comp_x_1", "comp_y_1"]

    config_dict = load_app_config_dict(config_file_path=config_file_path)

    components = component_factory.build_config(config_dict=config_dict, component_names=component_names)

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
def test_non_existing_reference(config_file_path: Path, component_factory: ComponentFactory):
    component_names = ["comp_x_1", "comp_y_1"]

    config_dict = load_app_config_dict(config_file_path=config_file_path)

    with pytest.raises(KeyError):
        component_factory.build_config(config_dict=config_dict, component_names=component_names)


@pytest.mark.parametrize(
    "config_file_path",
    [
        Path("tests/config/test_configs/config_hierarchical_list_component.yaml"),
    ],
)
def test_hierarchical_component_instantiation(config_file_path: Path, component_factory: ComponentFactory):
    component_names = ["comp_y_1"]

    config_dict = load_app_config_dict(config_file_path=config_file_path)

    components = component_factory.build_config(config_dict=config_dict, component_names=component_names)

    assert isinstance(components["comp_y_1"].multi_dependency[0], ComponentW)
    assert isinstance(components["comp_y_1"].multi_dependency[1], ComponentV)
    assert isinstance(components["comp_y_1"], ComponentY)


@pytest.mark.parametrize(
    "config_file_path",
    [
        Path("tests/config/test_configs/config_hierarchical_list_component.yaml"),
    ],
)
def test_component_filter(config_file_path: Path, component_factory: ComponentFactory):
    component_names = ["comp_y_1"]

    config_dict = load_app_config_dict(config_file_path=config_file_path)

    components = component_factory.build_config(config_dict=config_dict, component_names=component_names)
    assert "comp_y_1" in components

    component_names += "abc"
    with pytest.raises(KeyError):
        components = component_factory.build_config(config_dict=config_dict, component_names=component_names)


@pytest.mark.parametrize(
    "config_file_path",
    [
        Path("tests/config/test_configs/config_single_component.yaml"),
    ],
)
def test_single_component(config_file_path: Path, component_factory: ComponentFactory):
    component_names = ["custom_comp_1"]

    config_dict = load_app_config_dict(config_file_path=config_file_path)

    components = component_factory.build_config(config_dict=config_dict, component_names=component_names)
    assert "custom_comp_1" in components
