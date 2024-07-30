from typing import Any, Dict, List, Tuple, Union

import torch.nn as nn
from torchdistx.deferred_init import deferred_init

from modalities.component_instantiation.hierarchical_instantiation.custom_component_builder_if import ComponentBuilderIF
from modalities.component_instantiation.hierarchical_instantiation.hierarchical_instantiation import (
    HierarchicalInstantiation,
)
from modalities.component_instantiation.registry.registry import Registry


class DeferredInitModelBuilder(ComponentBuilderIF):
    def __init__(self, registry: Registry):
        self.hierarchical_instantiation = HierarchicalInstantiation(
            registry=registry, top_level_component_to_custom_factory={}
        )

    def build_component(
        self,
        current_component_config: Union[Dict, List, Any],
        component_config: Union[Dict, List, Any],
        top_level_components: Dict[str, Any],
        traversal_path: List,
    ) -> Tuple[nn.Module,]:
        fn = self.hierarchical_instantiation.build_components_recursively
        args = dict(
            current_component_config=current_component_config,
            component_config=component_config,
            top_level_components=top_level_components,
            traversal_path=traversal_path,
            allow_reference_config=False,
        )
        module = deferred_init.deferred_init(fn, **args)
        entity_key = traversal_path[-1]
        top_level_components[entity_key] = module
        return module, top_level_components
