from typing import Annotated, Any

import torch
import torch.nn as nn
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Sampler
from torch.utils.data.dataset import Dataset

from modalities.checkpointing.checkpoint_loading import CheckpointLoadingIF
from modalities.checkpointing.checkpoint_saving import CheckpointSaving, CheckpointSavingExecutionABC
from modalities.checkpointing.checkpoint_saving_strategies import CheckpointSavingStrategyIF
from modalities.dataloader.dataloader import LLMDataLoader
from modalities.inference.text.inference_component import TextInferenceComponent
from modalities.loops.training.gradient_clipping.gradient_clipper import GradientClipperIF
from modalities.loss_functions import Loss
from modalities.messaging.messages.payloads import BatchProgressUpdate, StepState
from modalities.messaging.publishers.publisher import MessagePublisherIF
from modalities.messaging.subscribers.subscriber import MessageSubscriberIF
from modalities.models.gpt2.collator import CollateFnIF
from modalities.tokenization.tokenizer_wrapper import TokenizerWrapper


class PydanticThirdPartyTypeIF:
    def __init__(self, third_party_type):
        self.third_party_type = third_party_type

    def __get_pydantic_core_schema__(
        self,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        # see: https://docs.pydantic.dev/latest/concepts/types/#handling-third-party-types
        return core_schema.json_or_python_schema(
            json_schema=core_schema.is_instance_schema(self.third_party_type),
            python_schema=core_schema.is_instance_schema(self.third_party_type),
            # serialization=core_schema.plain_serializer_function_ser_schema(
            #     lambda instance: instance.x
            # ),
        )


PydanticCheckpointSavingIFType = Annotated[CheckpointSaving, PydanticThirdPartyTypeIF(CheckpointSaving)]
PydanticCheckpointLoadingIFType = Annotated[CheckpointLoadingIF, PydanticThirdPartyTypeIF(CheckpointLoadingIF)]
PydanticCheckpointSavingStrategyIFType = Annotated[
    CheckpointSavingStrategyIF, PydanticThirdPartyTypeIF(CheckpointSavingStrategyIF)
]
PydanticCheckpointSavingExecutionIFType = Annotated[
    CheckpointSavingExecutionABC, PydanticThirdPartyTypeIF(CheckpointSavingExecutionABC)
]
PydanticPytorchModuleType = Annotated[nn.Module, PydanticThirdPartyTypeIF(nn.Module)]
PydanticTokenizerIFType = Annotated[TokenizerWrapper, PydanticThirdPartyTypeIF(TokenizerWrapper)]
PydanticDatasetIFType = Annotated[Dataset, PydanticThirdPartyTypeIF(Dataset)]
PydanticSamplerIFType = Annotated[Sampler, PydanticThirdPartyTypeIF(Sampler)]
PydanticCollateFnIFType = Annotated[CollateFnIF, PydanticThirdPartyTypeIF(CollateFnIF)]
PydanticLLMDataLoaderIFType = Annotated[LLMDataLoader, PydanticThirdPartyTypeIF(LLMDataLoader)]
PydanticOptimizerIFType = Annotated[Optimizer, PydanticThirdPartyTypeIF(Optimizer)]
PydanticLRSchedulerIFType = Annotated[LRScheduler, PydanticThirdPartyTypeIF(LRScheduler)]
PydanticLossIFType = Annotated[Loss, PydanticThirdPartyTypeIF(Loss)]
PydanticMessageSubscriberIFType = Annotated[MessageSubscriberIF, PydanticThirdPartyTypeIF(MessageSubscriberIF)]
PydanticPytorchDeviceType = Annotated[torch.device, PydanticThirdPartyTypeIF(torch.device)]
PydanticTextInferenceComponentType = Annotated[TextInferenceComponent, PydanticThirdPartyTypeIF(TextInferenceComponent)]
PydanticGradientClipperIFType = Annotated[GradientClipperIF, PydanticThirdPartyTypeIF(GradientClipperIF)]
PydanticMessagePublisherIFType = Annotated[MessagePublisherIF, PydanticThirdPartyTypeIF(MessagePublisherIF)]
PydanticBatchProgressUpdatePublisherIFType = Annotated[
    MessagePublisherIF[BatchProgressUpdate], PydanticThirdPartyTypeIF(MessagePublisherIF[BatchProgressUpdate])
]
PydanticStepStatePublisherIFType = Annotated[
    MessagePublisherIF[StepState], PydanticThirdPartyTypeIF(MessagePublisherIF[StepState])
]
