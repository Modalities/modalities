from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

from pydantic import BaseModel

from modalities.component_instantiation.hierarchical_instantiation.custom_component_builder_if import ComponentBuilderIF
from modalities.component_instantiation.registry.registry import Registry
from modalities.exceptions import HierachicalInstantiationError

BaseModelChild = TypeVar("BaseModelChild", bound=BaseModel)


class HierarchicalInstantiation:
    def __init__(
        self,
        registry: Registry,
        top_level_component_to_custom_factory: Optional[Dict[Tuple[str, str], ComponentBuilderIF]] = None,
    ):
        """_summary_

        Args:
            registry (Registry): _description_
            top_level_component_to_custom_factory (Optional[Dict[Tuple[str, str], ComponentBuilderIF]]): _description_.
                Defaults to None.
        """
        self.registry = registry
        if top_level_component_to_custom_factory is None:
            top_level_component_to_custom_factory = {}

        self.top_level_component_to_custom_factory = top_level_component_to_custom_factory

    def build_components_recursively(
        self,
        current_component_config: Union[Dict, List, Any],
        component_config: Union[Dict, List, Any],
        top_level_components: Dict[str, Any],
        traversal_path: List,
        allow_reference_config: bool = True,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """_summary_

        Args:
            current_component_config (Union[Dict, List, Any]): _description_
            component_config (Union[Dict, List, Any]): _description_
            top_level_components (Dict[str, Any]): Dictionary that contains the top-level components that
                have been built already. Initially empty.
            traversal_path (List): List of keys reflecting the traversal path within the config.
                Initially empty list.

        Returns:
            Any: _description_
        """
        # build sub components first via recursion
        if isinstance(current_component_config, dict):
            # if the entities are already top level components, we return the component,
            # as it must have been built already via a referencing component
            if len(traversal_path) > 0 and traversal_path[-1] in top_level_components:
                entity_key = traversal_path[-1]
                return top_level_components[entity_key], top_level_components

            # check if we have a custom instantiation config for this component
            if HierarchicalInstantiation._is_custom_component_config(
                config_dict=current_component_config,
                top_level_component_to_custom_factory=self.top_level_component_to_custom_factory,
            ):
                if len(traversal_path) > 1:
                    raise HierachicalInstantiationError(
                        f"Could not instantiate component {traversal_path[-1]}. Only top-level components allow"
                        " for custom instantiation routines."
                    )
                component_key = current_component_config["component_key"]
                variant_key = current_component_config["variant_key"]

            # if it is not a component that has been built already or a custom component, we need to build it.
            # We first traverse the config for possible sub components that need to build beforehand.
            materialized_component_config = {}
            for sub_entity_key, sub_component_config_dict in current_component_config.items():
                materialized_component_config[sub_entity_key], top_level_components = self.build_components_recursively(
                    current_component_config=sub_component_config_dict,
                    component_config=component_config,
                    top_level_components=top_level_components,
                    traversal_path=traversal_path + [sub_entity_key],
                    allow_reference_config=allow_reference_config,
                )
            # After building all the sub components, we can now build the actual component
            # if the config is component_config then we instantiate the component
            if HierarchicalInstantiation._is_component_config(config_dict=current_component_config):
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
                print(" -> ".join(traversal_path) + ":", component)

                # if the component is a top level component, then we add it to the top level components dictionary
                # to make sure that we don't build it again. Building it again would mean that we work by-value
                # instead of by reference.
                if len(traversal_path) == 1:
                    entity_key = traversal_path[-1]
                    top_level_components[entity_key] = component
                return component, top_level_components

            # if the config is a reference_config then check if it exists and if not, we build it
            if HierarchicalInstantiation._is_reference_config(config_dict=current_component_config):
                if not allow_reference_config:
                    raise ValueError("At least one of the components ")
                referenced_entity_key = current_component_config["instance_key"]
                if referenced_entity_key not in top_level_components:
                    materialized_referenced_component, top_level_components = self.build_components_recursively(
                        current_component_config=component_config[referenced_entity_key],
                        component_config=component_config,
                        top_level_components=top_level_components,
                        traversal_path=[referenced_entity_key],
                        allow_reference_config=allow_reference_config,
                    )
                    # we add the newly build reference config to the top level components dict
                    # so that we don't instantiate it again when we reach the respective component config
                    # in the subsequent config traversal
                    top_level_components[referenced_entity_key] = materialized_referenced_component
                print(" -> ".join(traversal_path) + ": ", f"--ref--> {top_level_components[referenced_entity_key]}")
                return top_level_components[referenced_entity_key], top_level_components

            return materialized_component_config, top_level_components

        elif isinstance(current_component_config, list):
            materialized_component_configs = []
            for sub_entity_key, sub_component_config in enumerate(current_component_config):
                materialized_component_config, top_level_components = self.build_components_recursively(
                    current_component_config=sub_component_config,
                    component_config=component_config,
                    top_level_components=top_level_components,
                    traversal_path=traversal_path + [str(sub_entity_key)],
                    allow_reference_config=allow_reference_config,
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
    def _is_custom_component_config(
        config_dict: Dict, top_level_component_to_custom_factory: Optional[Dict[Tuple[str, str], Callable]]
    ) -> bool:
        is_component_config = "component_key" in config_dict.keys() and "variant_key" in config_dict.keys()
        is_custom_component_config = (
            is_component_config
            and (config_dict["component_key"], config_dict["variant_key"]) in top_level_component_to_custom_factory
        )
        return is_custom_component_config

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
        component_config_dict = HierarchicalInstantiation.base_model_to_dict(component_config)
        component = component_type(**component_config_dict)
        return component

    @staticmethod
    def base_model_to_dict(base_model: BaseModel) -> Dict:
        # converts top level structure of base_model into dictionary while maintaining substructure
        output = {}
        for name, _ in base_model.model_fields.items():
            value = getattr(base_model, name)
            output[name] = value
        return output
