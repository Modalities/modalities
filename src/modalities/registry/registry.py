from typing import Dict, Tuple, Type


class Registry:
    def __init__(self) -> None:
        # maps component_key -> variant_key -> entity = (component, config)
        self._registry_dict: Dict[str, Dict[str, Type]] = {}

    def add_entity(self, component_key: str, variant_key: str, entity: Tuple[Type, Type]) -> None:
        if component_key not in self._registry_dict:
            self._registry_dict[component_key] = {}
        self._registry_dict[component_key][variant_key] = entity

    def get_component(self, component_key: str, variant_key: str) -> Type:
        try:
            return self._registry_dict[component_key][variant_key][0]
        except KeyError as e:
            raise ValueError(f"[{component_key}][{variant_key}] are not valid keys in registry") from e
        except IndexError as e:
            raise ValueError(f"0 is not a valid index in registry[{component_key}][{variant_key}]") from e

    def get_config(self, component_key: str, variant_key: str) -> Type:
        try:
            return self._registry_dict[component_key][variant_key][1]
        except KeyError as e:
            raise ValueError(f"[{component_key}][{variant_key}] are not valid keys in registry") from e
        except IndexError as e:
            raise ValueError(f"0 is not a valid index in registry[{component_key}][{variant_key}]") from e
