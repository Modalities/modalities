from pathlib import Path
from typing import Annotated

import pytest
from pydantic import BaseModel

from modalities.config.component_factory import ComponentFactory
from modalities.config.config import load_app_config_dict
from modalities.config.pydanctic_if_types import PydanticThirdPartyTypeIF
from modalities.registry.components import ComponentEntity
from modalities.registry.registry import Registry
from tests.config.components import ComponentV, ComponentW, ComponentX, ComponentY
from tests.config.configs import CompVConfig, CompWConfig, CompXConfig, CompYConfig


@pytest.fixture(scope="function")
def component_factory() -> ComponentFactory:
    components = [
        ComponentEntity("COMP_V", "default", ComponentV, CompVConfig),
        ComponentEntity("COMP_W", "default", ComponentW, CompWConfig),
        ComponentEntity("COMP_X", "default", ComponentX, CompXConfig),
        ComponentEntity("COMP_Y", "default", ComponentY, CompYConfig),
    ]

    registry = Registry(components=components)
    component_factory = ComponentFactory(registry=registry)
    return component_factory


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

    components = component_factory._build_config(
        config_dict=config_dict, component_names_required=component_names, component_names_optional=[]
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
def test_non_existing_reference(config_file_path: Path, component_factory: ComponentFactory):
    component_names = ["comp_x_1", "comp_y_1"]

    config_dict = load_app_config_dict(config_file_path=config_file_path)

    with pytest.raises(KeyError):
        component_factory._build_config(
            config_dict=config_dict, component_names_required=component_names, component_names_optional=[]
        )


@pytest.mark.parametrize(
    "config_file_path",
    [
        Path("tests/config/test_configs/config_hierarchical_list_component.yaml"),
    ],
)
def test_hierarchical_component_instantiation(config_file_path: Path, component_factory: ComponentFactory):
    component_names = ["comp_y_1"]

    config_dict = load_app_config_dict(config_file_path=config_file_path)

    components = component_factory._build_config(
        config_dict=config_dict, component_names_required=component_names, component_names_optional=[]
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
def test_component_filter(config_file_path: Path, component_factory: ComponentFactory):
    component_names = ["comp_y_1"]

    config_dict = load_app_config_dict(config_file_path=config_file_path)

    components = component_factory._build_config(
        config_dict=config_dict, component_names_required=component_names, component_names_optional=[]
    )
    assert "comp_y_1" in components

    component_names += "abc"
    with pytest.raises(KeyError):
        components = component_factory._build_config(
            config_dict=config_dict, component_names_required=component_names, component_names_optional=[]
        )


@pytest.mark.parametrize(
    "config_file_path",
    [
        Path("tests/config/test_configs/config_single_component.yaml"),
    ],
)
def test_single_component(config_file_path: Path, component_factory: ComponentFactory):
    component_names = ["custom_comp_1"]

    config_dict = load_app_config_dict(config_file_path=config_file_path)

    components = component_factory._build_config(
        config_dict=config_dict, component_names_required=component_names, component_names_optional=[]
    )
    assert "custom_comp_1" in components


@pytest.mark.parametrize(
    "config_file_path",
    [
        Path("tests/config/test_configs/config_multiple_top_level_components_with_references.yaml"),
    ],
)
def test_not_referencing_when_dictionary_keys_match(config_file_path: Path, component_factory: ComponentFactory):
    class ComponentModel(BaseModel):
        val_x: Annotated[ComponentX, PydanticThirdPartyTypeIF(ComponentX)]
        single_dependency: Annotated[ComponentW, PydanticThirdPartyTypeIF(ComponentW)]
        comp_x_1: Annotated[ComponentX, PydanticThirdPartyTypeIF(ComponentX)]

    # we make sure that even if the keys in the dict are the same as the keys of top-level components
    # we still don't reference these top level components
    # (we only want to reference top-level components via instance_key and pass_type)
    components: ComponentModel = component_factory.build_components(
        config_dict=load_app_config_dict(config_file_path), components_model_type=ComponentModel
    )
    assert components.val_x.val_x == "val_x -> config -> val_x"
    assert components.comp_x_1.val_x == "comp_x_1 -> config -> val_x"

    assert components.val_x.single_dependency is not components.comp_x_1.single_dependency
    assert components.val_x.single_dependency.val_w == components.comp_x_1.single_dependency.val_w
    assert components.single_dependency is not components.comp_x_1.single_dependency
    assert components.single_dependency.val_w == "single_dependency -> config -> val_w"


@pytest.mark.parametrize(
    "config_file_path",
    [
        Path("tests/config/test_configs/config_multiple_top_level_components_with_references.yaml"),
    ],
)
def test_referencing_with_component_model(config_file_path: Path, component_factory: ComponentFactory):
    class ComponentModel(BaseModel):
        val_x: Annotated[ComponentX, PydanticThirdPartyTypeIF(ComponentX)]
        comp_x_1: Annotated[ComponentX, PydanticThirdPartyTypeIF(ComponentX)]
        comp_x_2: Annotated[ComponentX, PydanticThirdPartyTypeIF(ComponentX)]
        comp_w_1: Annotated[ComponentW, PydanticThirdPartyTypeIF(ComponentW)]
        comp_x_3: Annotated[ComponentX, PydanticThirdPartyTypeIF(ComponentX)]

    components: ComponentModel = component_factory.build_components(
        config_dict=load_app_config_dict(config_file_path), components_model_type=ComponentModel
    )
    assert components.comp_x_2.single_dependency is components.comp_w_1
    assert components.comp_x_3.single_dependency is components.comp_w_1
    assert components.comp_x_1.single_dependency is not components.comp_w_1


class TestComponentFactory:
    @pytest.fixture
    def component_factory(self):
        # Create a ComponentFactory instance with a dummy registry
        return ComponentFactory(registry=None)

    def test_assert_valid_config_keys_with_valid_keys(self, component_factory):
        # Define a dummy component config type
        class DummyConfig(BaseModel):
            required_key: str
            optional_key: str = "default"

        # Create a valid config dictionary
        config_dict = {
            "required_key": "value",
            "optional_key": "value",
        }

        # Call the _assert_valid_config_keys method
        component_factory._assert_valid_config_keys(
            component_key="dummy",
            variant_key="default",
            config_dict=config_dict,
            component_config_type=DummyConfig,
        )

        # No exception should be raised

    def test_assert_valid_config_keys_with_invalid_keys(self, component_factory):
        # Define a dummy component config type
        class DummyConfig(BaseModel):
            required_key: str
            optional_key: str = "default"

        # Create an invalid config dictionary with an extra key
        config_dict = {
            "required_key": "value",
            "optional_key": "value",
            "extra_key": "value",
        }

        # Call the _assert_valid_config_keys method and expect a ValueError
        with pytest.raises(ValueError):
            component_factory._assert_valid_config_keys(
                component_key="dummy",
                variant_key="default",
                config_dict=config_dict,
                component_config_type=DummyConfig,
            )

        # Create a valid config dictionary
        valid_config_dict = {
            "required_key": "value",
            "optional_key": "value",
        }

        # Call the _assert_valid_config_keys method and expect no exception
        component_factory._assert_valid_config_keys(
            component_key="dummy",
            variant_key="default",
            config_dict=valid_config_dict,
            component_config_type=DummyConfig,
        )
