from typing import Any, Dict, List, Type, TypeVar, Union

from pydantic import BaseModel

from modalities.registry.registry import Registry
from modalities.util import print_rank_0

BaseModelChild = TypeVar("BaseModelChild", bound=BaseModel)


class ComponentFactory:
    """Factory class to build the components from a config dictionary."""

    def __init__(self, registry: Registry) -> None:
        """Initializes the ComponentFactory with a registry.

        Args:
            registry (Registry): Registry object to get the component and config classes.
        """
        self.registry = registry

    def build_components(self, config_dict: Dict, components_model_type: Type[BaseModelChild]) -> BaseModelChild:
        """Builds the components from a config dictionary. All components specified in `components_model_type`
        are built from the config dictionary in a recursive manner.

        Args:
            config_dict (Dict): Dictionary with the configuration of the components.
            components_model_type (Type[BaseModelChild]): Base model type defining the components to be build.

        Returns:
            BaseModelChild: Instance of the components_model_type with the built components.
        """
        component_names = list(components_model_type.model_fields.keys())
        component_dict = self._build_config(config_dict=config_dict, component_names=component_names)
        components = components_model_type(**component_dict)
        return components

    def _build_config(self, config_dict: Dict, component_names: List[str]) -> Dict[str, Any]:
        component_dict_filtered = {name: config_dict[name] for name in component_names}
        components, _ = self._build_component(
            current_component_config=component_dict_filtered,
            component_config=config_dict,
            top_level_components={},
            traversal_path=[],
        )
        return components

    def _build_component(
        self,
        current_component_config: Union[Dict, List, Any],
        component_config: Union[Dict, List, Any],
        top_level_components: Dict[str, Any],
        traversal_path: List,
    ) -> Any:
        # build sub components first via recursion
        if isinstance(current_component_config, dict):
            # if the entities are top level components, we return the component,
            # as it must have been built already via a referencing component
            if len(traversal_path) > 0 and traversal_path[-1] in top_level_components:
                entity_key = traversal_path[-1]
                return top_level_components[entity_key], top_level_components
            # if it is not a component that has been built already, we need to build it.
            # We first traverse the config for possible sub components that need to build beforehand.
            materialized_component_config = {}
            for sub_entity_key, sub_component_config_dict in current_component_config.items():
                materialized_component_config[sub_entity_key], top_level_components = self._build_component(
                    current_component_config=sub_component_config_dict,
                    component_config=component_config,
                    top_level_components=top_level_components,
                    traversal_path=traversal_path + [sub_entity_key],
                )
            # After building all the sub components, we can now build the actual component
            # if the config is component_config then we instantiate the component
            if ComponentFactory._is_component_config(config_dict=current_component_config):
                # instantiate component config
                component_key = current_component_config["component_key"]
                variant_key = current_component_config["variant_key"]
                current_component_config = self._instantiate_component_config(
                    component_key=component_key,
                    variant_key=variant_key,
                    config_dict=materialized_component_config["config"],
                )
                # instantiate component
                component = self._instantiate_component(
                    component_key=component_key, variant_key=variant_key, component_config=current_component_config
                )
                print_rank_0(f"Instantiated {type(component)}: {' -> '.join(traversal_path)}")

                # if the component is a top level component, then we add it to the top level components dictionary
                # to make sure that we don't build it again. Building it again would mean that we work by-value
                # instead of by reference.
                if len(traversal_path) == 1:
                    entity_key = traversal_path[-1]
                    top_level_components[entity_key] = component
                return component, top_level_components

            # if the config is a reference_config then check if it exists and if not, we build it
            if ComponentFactory._is_reference_config(config_dict=current_component_config):
                referenced_entity_key = current_component_config["instance_key"]
                if referenced_entity_key not in top_level_components:
                    materialized_referenced_component, top_level_components = self._build_component(
                        current_component_config=component_config[referenced_entity_key],
                        component_config=component_config,
                        top_level_components=top_level_components,
                        traversal_path=[referenced_entity_key],
                    )
                    # we add the newly build reference config to the top level components dict
                    # so that we don't instantiate it again when we reach the respective component config
                    # in the subsequent config traversal
                    top_level_components[referenced_entity_key] = materialized_referenced_component
                return top_level_components[referenced_entity_key], top_level_components

            return materialized_component_config, top_level_components

        elif isinstance(current_component_config, list):
            materialized_component_configs = []
            for sub_entity_key, sub_component_config in enumerate(current_component_config):
                materialized_component_config, top_level_components = self._build_component(
                    current_component_config=sub_component_config,
                    component_config=component_config,
                    top_level_components=top_level_components,
                    traversal_path=traversal_path + [str(sub_entity_key)],
                )
                materialized_component_configs.append(materialized_component_config)
            return materialized_component_configs, top_level_components

        else:
            # we return the raw sub config if the sub config is not a dictionary or a list
            # i.e., just a "scalar" value (e.g., string, int, etc.), since we don't have to build it.
            return current_component_config, top_level_components

    @staticmethod
    def _is_component_config(config_dict: Dict) -> bool:
        # TODO instead of field checks, we should introduce an enum for the config type.
        return "component_key" in config_dict.keys()

    @staticmethod
    def _is_reference_config(config_dict: Dict) -> bool:
        # TODO instead of field checks, we should introduce an enum for the config type.
        return {"instance_key", "pass_type"} == config_dict.keys()

    def _instantiate_component_config(self, component_key: str, variant_key: str, config_dict: Dict) -> BaseModel:
        component_config_type: Type[BaseModel] = self.registry.get_config(component_key, variant_key)
        self._assert_valid_config_keys(
            component_key=component_key,
            variant_key=variant_key,
            config_dict=config_dict,
            component_config_type=component_config_type,
        )
        comp_config = component_config_type(**config_dict, strict=True)
        return comp_config

    def _assert_valid_config_keys(
        self, component_key: str, variant_key: str, config_dict: Dict, component_config_type: Type[BaseModelChild]
    ) -> None:
        required_keys = []
        optional_keys = []
        for key, field in component_config_type.model_fields.items():
            if field.is_required():
                required_keys.append(key)
            else:
                optional_keys.append(key)

        invalid_keys = []
        for key in config_dict.keys():
            if key not in required_keys and key not in optional_keys:
                invalid_keys.append(key)
        if len(invalid_keys) > 0:
            message = f"Invalid keys {invalid_keys} for config `{component_key}.{variant_key}`"
            message += f" of type {component_config_type}:\n{config_dict}\n"
            message += f"Required keys: {required_keys}\nOptional keys: {optional_keys}"
            raise ValueError(message)

    def _instantiate_component(self, component_key: str, variant_key: str, component_config: BaseModel) -> Any:
        component_type: Type = self.registry.get_component(component_key, variant_key)
        component_config_dict = self._base_model_to_dict(component_config)
        component = component_type(**component_config_dict)
        return component

    @staticmethod
    def _base_model_to_dict(base_model: BaseModel) -> Dict:
        # converts top level structure of base_model into dictionary while maintaining substructure
        output = {}
        for name, _ in base_model.model_fields.items():
            value = getattr(base_model, name)
            output[name] = value
        return output
