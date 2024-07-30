from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union


class ComponentBuilderIF(ABC):
    @abstractmethod
    def build_component(
        self,
        current_component_config: Union[Dict, List, Any],
        component_config: Union[Dict, List, Any],
        top_level_components: Dict[str, Any],
        traversal_path: List,
    ) -> Any:
        raise NotImplementedError
