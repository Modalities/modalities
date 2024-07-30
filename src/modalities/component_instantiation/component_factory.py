from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar

from pydantic import BaseModel

from modalities.component_instantiation.hierarchical_instantiation.custom_component_builder_if import ComponentBuilderIF
from modalities.component_instantiation.hierarchical_instantiation.custom_component_builders import (
    DeferredInitModelBuilder,
)
from modalities.component_instantiation.hierarchical_instantiation.hierarchical_instantiation import (
    HierarchicalInstantiation,
)
from modalities.component_instantiation.registry.registry import Registry

BaseModelChild = TypeVar("BaseModelChild", bound=BaseModel)


class ComponentFactory:
    def __init__(
        self,
        registry: Registry,
        top_level_component_to_custom_factory: Optional[Dict[Tuple[str, str], Callable]] = None,
    ) -> None:
        """_summary_

        Args:
            registry (Registry): _description_
            top_level_component_to_custom_factory (Optional[Dict[Tuple[str, str], Callable]]): _description_.
                Defaults to None.
        """
        if top_level_component_to_custom_factory is None:
            top_level_component_to_custom_factory: Dict[Tuple[str, str], ComponentBuilderIF] = {
                ("model", "deferred_init"): DeferredInitModelBuilder(registry=registry)
            }
        self.hierarchical_instantiation = HierarchicalInstantiation(
            registry=registry, top_level_component_to_custom_factory=top_level_component_to_custom_factory
        )

    def build_components(self, config_dict: Dict, components_model_type: Type[BaseModelChild]) -> BaseModelChild:
        """_summary_

        Args:
            config_dict (Dict): _description_
            components_model_type (Type[BaseModelChild]): _description_

        Returns:
            BaseModelChild: _description_
        """
        component_names = list(components_model_type.model_fields.keys())

        component_dict = self.build_components_from_names(config_dict=config_dict, component_names=component_names)

        components = components_model_type(**component_dict)
        return components

    def build_components_from_names(self, config_dict: Dict, component_names: List[str]) -> Dict[str, Any]:
        component_dict_filtered = {name: config_dict[name] for name in component_names}
        component_dict, _ = self.hierarchical_instantiation.build_components_recursively(
            current_component_config=component_dict_filtered,
            component_config=config_dict,
            top_level_components={},
            traversal_path=[],
        )
        return component_dict
