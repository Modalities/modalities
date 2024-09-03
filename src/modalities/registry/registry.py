from dataclasses import asdict
from typing import Dict, List, Optional, Tuple, Type

from pydantic import BaseModel

from modalities.registry.components import ComponentEntity

Entity = Tuple[Type, Type[BaseModel]]


class Registry:
    """Registry class to store the components and their config classes."""

    def __init__(self, components: Optional[List[ComponentEntity]] = None) -> None:
        """Initializes the Registry class with an optional list of components.

        Args:
            components (List[ComponentEntity], optional): List of components to
                intialize the registry with . Defaults to None.
        """
        # maps component_key -> variant_key -> entity = (component, config)
        self._registry_dict: Dict[str, Dict[str, Entity]] = {}
        if components is not None:
            for component in components:
                self.add_entity(**asdict(component))

    def add_entity(
        self, component_key: str, variant_key: str, component_type: Type, component_config_type: Type[BaseModel]
    ) -> None:
        """Adds a component to the registry.

        The registry has a two-level dictionary structure, where the first level is the component_key
        and the second level is the variant_key. The component_key is used to identify the component type,
        whereas the variant_key is used to identify the component variant. For instance, for a GPT 2 model the
        component key could be "model" and the variant key could be "gpt2".

        Args:
            component_key (str): Key to identify the component type.
            variant_key (str): Variant key to identify the component.
            component_type (Type): Type of the component.
            component_config_type (Type[BaseModel]): Type of the component config.
        """
        if component_key not in self._registry_dict:
            self._registry_dict[component_key] = {}
        self._registry_dict[component_key][variant_key] = (component_type, component_config_type)

    def get_component(self, component_key: str, variant_key: str) -> Type:
        """Returns the component type for a given component_key and variant_key.

        Args:
            component_key (str): Component key to identify the component type.
            variant_key (str): Variant key to identify the component variant.

        Raises:
            ValueError: Raises a ValueError if the component_key or variant_key are not valid keys in the registry.

        Returns:
            Type: Component type.
        """
        entity = self._get_entity(component_key, variant_key)
        try:
            return entity[0]
        except IndexError as e:
            raise ValueError(f"0 is not a valid index in registry[{component_key}][{variant_key}]") from e

    def get_config(self, component_key: str, variant_key: str) -> Type[BaseModel]:
        """Returns the config type for a given component_key and variant_key.

        Args:
            component_key (str): Component key to identify the component type.
            variant_key (str): Variant key to identify the component variant.

        Raises:
            ValueError: Raises a ValueError if the component_key or variant_key are not valid keys in the registry.

        Returns:
            Type[BaseModel]: Config type
        """
        entity = self._get_entity(component_key, variant_key)
        try:
            return entity[1]
        except IndexError as e:
            raise ValueError(f"1 is not a valid index in registry[{component_key}][{variant_key}]") from e

    def _get_entity(self, component_key: str, variant_key: str) -> Entity:
        try:
            return self._registry_dict[component_key][variant_key]
        except KeyError as e:
            raise ValueError(f"[{component_key}][{variant_key}] are not valid keys in registry") from e
