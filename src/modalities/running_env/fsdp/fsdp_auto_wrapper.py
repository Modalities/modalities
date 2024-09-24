import functools
import logging
from abc import ABC, abstractmethod
from typing import Callable

import torch.nn as nn
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from modalities.config.lookup_enum import LookupEnum
from modalities.util import get_module_class_from_name, print_rank_0


class FSDPAutoWrapFactoryIF(ABC):
    @abstractmethod
    def get_auto_wrap_policy(self) -> Callable:
        raise NotImplementedError


class FSDPTransformerAutoWrapPolicyFactory(FSDPAutoWrapFactoryIF):
    def __init__(self, model: nn.Module, block_names: list[str]) -> None:
        # TODO it's problematic that we store the model in-memory here. Might get too large in RAM...
        self.model = model
        self.block_names = block_names

    @staticmethod
    def _get_fsdp_blocks_from_block_names(model: nn.Module, block_names: list[str]) -> list[nn.Module]:
        fsdp_block_types = []
        for cls_block_name in block_names:
            # TODO FullyShardedDataParallelPlugin from Accelerate uses string matching to find the correct
            # block class. In the long-term we should implmement this ourselves in a robuster fashion.
            block_type = get_module_class_from_name(model, cls_block_name)

            if block_type is None:
                raise ValueError(f"Could not find block with name {cls_block_name} in model")
            fsdp_block_types.append(block_type)
        return fsdp_block_types

    def get_auto_wrap_policy(self) -> Callable:
        transformer_layer_cls = self._get_fsdp_blocks_from_block_names(model=self.model, block_names=self.block_names)
        logging.info(f"Wrapped layer classes: {transformer_layer_cls}\n")
        print_rank_0(f"\nWrapped layer classes: {transformer_layer_cls}\n")

        if len(transformer_layer_cls) == 0:
            raise ValueError("No FSDP blocks found in model")

        auto_wrapper_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                *transformer_layer_cls,
            },
        )
        return auto_wrapper_policy


class FSDPAutoWrapFactoryTypes(LookupEnum):
    FSDPTransformerAutoWrapPolicyFactory = FSDPTransformerAutoWrapPolicyFactory
