#!/usr/bin/env python3

from typing import Optional

from pydantic import FilePath

from modalities.config.component_factory import ComponentFactory
from modalities.config.config import ProcessGroupBackendType, load_app_config_dict
from modalities.config.instantiation_models import TextGenerationInstantiationModel
from modalities.inference.text.config import TextInferenceComponentConfig
from modalities.inference.text.inference_component import TextInferenceComponent
from modalities.registry.components import COMPONENTS
from modalities.registry.registry import Registry
from modalities.running_env.cuda_env import CudaEnv
from modalities.running_env.env_utils import is_running_with_torchrun


def generate_text(config_path: FilePath, registry: Optional[Registry] = None):
    config_dict = load_app_config_dict(config_path)
    if registry is None:
        registry = Registry(COMPONENTS)
    registry.add_entity(
        component_key="inference_component",
        variant_key="text",
        component_type=TextInferenceComponent,
        component_config_type=TextInferenceComponentConfig,
    )
    component_factory = ComponentFactory(registry=registry)

    if is_running_with_torchrun():
        with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
            components = component_factory.build_components(
                config_dict=config_dict,
                components_model_type=TextGenerationInstantiationModel,
            )

    else:
        components = component_factory.build_components(
            config_dict=config_dict,
            components_model_type=TextGenerationInstantiationModel,
        )
    text_inference_component = components.text_inference_component

    text_inference_component.run()
