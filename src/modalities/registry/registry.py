from typing import Dict, List, Optional, Tuple, Type

from pydantic import BaseModel

Entity = Tuple[Type, Type[BaseModel]]  # TODO: replace by BaseModel


class Registry:
    def __init__(self, components: Optional[List] = None) -> None:  # TODO: fix typing, introduce BaseModels
        # maps component_key -> variant_key -> entity = (component, config)
        self._registry_dict: Dict[str, Dict[str, Entity]] = {}
        if components is not None:
            for component in components:
                self.add_entity(*component)

    def add_entity(self, component_key: str, variant_key: str, entity: Entity) -> None:
        if component_key not in self._registry_dict:
            self._registry_dict[component_key] = {}
        self._registry_dict[component_key][variant_key] = entity

    def get_component(self, component_key: str, variant_key: str) -> Type:
        entity = self._get_entity(component_key, variant_key)
        try:
            return entity[0]
        except IndexError as e:
            raise ValueError(f"0 is not a valid index in registry[{component_key}][{variant_key}]") from e

    def get_config(self, component_key: str, variant_key: str) -> Type[BaseModel]:
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
