from typing import Dict, Type


class Registry:
    def __init__(self) -> None:
        # maps component_key -> variant_key -> component
        self._registry_dict: Dict[str, Dict[str, Type]] = {}

    def add_entity(self, entity_key: str, variant_key: str, entity: Type) -> None:
        if entity_key not in self._registry_dict:
            self._registry_dict[entity_key] = {}
        self._registry_dict[entity_key][variant_key] = entity

    def get_entity(self, component_key: str, variant_key: str) -> Type:
        return self._registry_dict[component_key][variant_key]
