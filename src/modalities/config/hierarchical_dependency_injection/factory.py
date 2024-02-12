from pathlib import Path
from typing import Any, Dict, List, Union

from omegaconf import OmegaConf
from pydantic import RootModel

from modalities.config.hierarchical_dependency_injection.configs import (
    CompConfigABC,
    CompVConfig,
    CompWConfig,
    CompXConfig,
    CompYConfig,
    CompZConfig,
    ReferenceConfig,
)


class ComponentFactory:
    @staticmethod
    def build_config(config_dict: Dict, config_types, component_names: List[str]) -> Dict[str, Any]:
        component_dict = ComponentFactory._build_components(
            config_dict=config_dict, config_types=config_types, component_names=component_names
        )
        return component_dict

    @staticmethod
    def _build_components(config_dict: Dict, config_types, component_names: List[str]) -> Dict[str, Any]:
        components, _ = ComponentFactory._build_component(
            current_component_config=config_dict,
            component_config=config_dict,
            top_level_components={},
            config_types=config_types,
            traversal_path=[],
        )
        return {name: components[name] for name in component_names}

    @staticmethod
    def _build_component(
        current_component_config: Union[Dict, List, Any],
        component_config: Union[Dict, List, Any],
        config_types,
        top_level_components: Dict[str, Any],
        traversal_path: List,
    ) -> Any:
        # build sub components first via recursion
        if isinstance(current_component_config, dict):
            # if the entities are top level components, we return the component,
            # as it must have been built already via a referencing component
            if len(traversal_path) > 0 and traversal_path[-1] in top_level_components:
                return top_level_components[traversal_path[-1]], top_level_components

            materialized_component_config = {}
            # before materializing the component, we need to materialize the sub components
            for sub_entity_key, sub_component_config_dict in current_component_config.items():
                materialized_component_config[sub_entity_key], top_level_components = ComponentFactory._build_component(
                    current_component_config=sub_component_config_dict,
                    component_config=component_config,
                    config_types=config_types,
                    top_level_components=top_level_components,
                    traversal_path=traversal_path + [sub_entity_key],
                )

            if len(traversal_path) == 0:
                return materialized_component_config, top_level_components

            # if the config is component_config then we instantiate the component
            if ComponentFactory._is_component_config(config_dict=current_component_config):
                # instantiate component config
                current_component_config = ComponentFactory._instantiate_component_config(
                    config_dict=materialized_component_config, config_types=config_types
                )
                # instantiate component
                component = ComponentFactory._instantiate_component(component_config=current_component_config)
                print(" -> ".join(traversal_path) + ":", component)
                if len(traversal_path) == 1:
                    entity_key = traversal_path[-1]
                    top_level_components[entity_key] = component
                return component, top_level_components

            # if the config is a reference_config then check if it exists and if not, we build it
            if ComponentFactory._is_reference_config(config_dict=current_component_config):
                referenced_entity_key = current_component_config["instance_key"]
                if referenced_entity_key not in top_level_components:
                    materialized_referenced_component, top_level_components = ComponentFactory._build_component(
                        current_component_config=component_config[referenced_entity_key],
                        component_config=component_config[referenced_entity_key],
                        config_types=config_types,
                        top_level_components=top_level_components,
                        traversal_path=[referenced_entity_key],
                    )
                    top_level_components[referenced_entity_key] = materialized_referenced_component
                print(" -> ".join(traversal_path) + ": ", f"--ref--> {top_level_components[referenced_entity_key]}")
                return top_level_components[referenced_entity_key], top_level_components

            return materialized_component_config, top_level_components

        elif isinstance(current_component_config, list):
            materialized_component_configs = []
            for sub_entity_key, sub_component_config in enumerate(current_component_config):
                materialized_component_config, top_level_components = ComponentFactory._build_component(
                    current_component_config=sub_component_config,
                    component_config=component_config,
                    config_types=config_types,
                    top_level_components=top_level_components,
                    traversal_path=traversal_path + [str(sub_entity_key)],
                )
                materialized_component_configs.append(materialized_component_config)
            return materialized_component_configs, top_level_components

        else:
            return current_component_config, top_level_components

    @staticmethod
    def _build_reference_component(
        current_component_config: Union[Dict, List, Any],
        component_config: Union[Dict, List, Any],
        config_types,
    ) -> Any:
        pass

    @staticmethod
    def _is_component_config(config_dict: Dict) -> bool:
        # TODO instead of field checks, we should introduce an enum for the config type.
        return "type_hint" in config_dict.keys()

    @staticmethod
    def _is_reference_config(config_dict: Dict) -> bool:
        # TODO instead of field checks, we should introduce an enum for the config type.
        return {"instance_key", "pass_type"} == config_dict.keys()

    @staticmethod
    def _instantiate_component_config(config_dict: Dict, config_types) -> CompConfigABC:
        comp_config = RootModel[config_types].model_validate(config_dict, strict=True).root
        return comp_config

    @staticmethod
    def _instantiate_component(component_config: CompConfigABC) -> Any:
        component_type = component_config.type_hint.value
        component = component_type(**component_config.model_dump(exclude=["type_hint"]))
        return component


if __name__ == "__main__":

    def load_app_config_dict(config_file_path: Path) -> Dict:
        cfg = OmegaConf.load(config_file_path)
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        return config_dict

    config_file_path = Path(
        "/raid/s3/opengptx/max_lue/modalities/src/modalities/config/hierarchical_dependency_injection/config.yaml"
    )
    comp_config_types = Union[CompVConfig, CompWConfig, CompXConfig, CompYConfig, CompZConfig, ReferenceConfig]
    component_names = ["comp_z_1", "comp_x_1", "comp_y_1"]

    config_dict = load_app_config_dict(config_file_path=config_file_path)

    components = ComponentFactory.build_config(
        config_dict=config_dict, config_types=comp_config_types, component_names=component_names
    )
    print(components)
