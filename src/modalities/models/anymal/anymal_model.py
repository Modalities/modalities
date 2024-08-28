from typing import Dict

import torch
from peft import LoraConfig, get_peft_model
from pydantic import BaseModel
from torch import nn

from modalities.config.lookup_enum import LookupEnum
from modalities.config.pydanctic_if_types import PydanticPytorchModuleType
from modalities.models.model import NNModel
from modalities.models.vision_transformer.vision_transformer_model import Perceiver, PerceiverConfig


class AnyMALTrainingStage(LookupEnum):
    MODALITY_ALIGNMENT = "modality_alignment"
    INSTRUCTION_TUNING = "instruction_tuning"


class AnyMALConfig(BaseModel):
    """

    Args:
        text_decoder (`PydanticPytorchModuleType`):
            a trained LLM
        prediction_key (`str`):
            key for the output logits of the model
        vision_encoder (`PydanticPytorchModuleType`), required if training an image-text model:
            a trained vision encoder (such as VisionTransformer or CLIPVisionModel
        vision_projector_config (`PerceiverConfig`), required if training an image-text model:
            config for Perceiver, which projects embeddings from the vision encoder
            to the text token embedding space
        audio_encoder (`PydanticPytorchModuleType`), required if training an audio-text model:
            a trained audio encoder such as AudioTransformer
        audio_encoder_n_embd (`int`):
            feature dimensionality of audio encoder
        audio_projector_config (`PerceiverConfig`), required if training an audio-text model:
            config for Perceiver, which projects embeddings from the audio encoder
            to the text token embedding space
        training_stage (`AnyMALTrainingStage`):
            training stage for the model. The first modality alignment stage trains the modality projector using
            captioned images/audio. The second instruction fine-tuning stage trains the modality projector
            and fine-tunes the LLM (using LoRA) using instructions, request and response.


    """

    text_decoder: PydanticPytorchModuleType
    prediction_key: str
    vision_encoder: PydanticPytorchModuleType = None
    vision_projector: PydanticPytorchModuleType = None
    audio_encoder: PydanticPytorchModuleType = None
    audio_encoder_n_embd: int = 512
    audio_projector_config: PerceiverConfig = None
    training_stage: AnyMALTrainingStage = AnyMALTrainingStage.MODALITY_ALIGNMENT


class AnyMAL(NNModel):
    """Implementation of the AnyMAL model for multimodal alignment and instruction tuning of LLMs.
    Based on the paper:AnyMAL: An Efficient and Scalable Any-Modality Augmented Language Model
    Link: https://arxiv.org/abs/2309.16058
    """

    def __init__(
        self,
        text_decoder: PydanticPytorchModuleType,
        prediction_key: str,
        vision_encoder: PydanticPytorchModuleType = None,
        vision_projector: PydanticPytorchModuleType = None,
        audio_encoder: PydanticPytorchModuleType = None,
        audio_encoder_n_embd: int = 512,
        audio_projector_config: PerceiverConfig = None,
        training_stage: AnyMALTrainingStage = AnyMALTrainingStage.MODALITY_ALIGNMENT,
        seed: int = None,
    ) -> None:
        super().__init__(seed=seed)
        self.prediction_key = prediction_key
        self.training_stage = training_stage

        if (vision_encoder is not None and audio_encoder is not None) or (
            vision_encoder is None and audio_encoder is None
        ):
            raise ValueError("Either a vision or audio encoder should be specified.")

        if vision_encoder is not None:
            if vision_projector is None:
                raise ValueError("Vision projector should not be None.")
            self.modality_prediction_key = vision_encoder.prediction_key
            self.modality_encoder = vision_encoder
            self.modality_projector = vision_projector
        elif audio_encoder is not None:
            if audio_projector_config is None:
                raise ValueError("Audio projector should not be None.")
            self.modality_prediction_key = audio_encoder.prediction_key
            self.modality_encoder = audio_encoder
            audio_projector_config.block_size = self.modality_encoder.block_size
            self.modality_projector = nn.Sequential(
                nn.Linear(audio_encoder_n_embd, audio_projector_config.n_embd),
                Perceiver(**dict(audio_projector_config)),
            )

        self.text_decoder = text_decoder

        # Freeze parameters and/or initialize LoRA based on training stage
        # currently does not work with FSDP
        for param in self.modality_encoder.parameters():
            param.requires_grad = False

        if training_stage == AnyMALTrainingStage.MODALITY_ALIGNMENT:
            for param in self.text_decoder.parameters():
                param.requires_grad = False
        elif training_stage == AnyMALTrainingStage.INSTRUCTION_TUNING:
            peft_config = LoraConfig(
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_attn", "k_attn", "v_attn", "c_proj", "c_fc"],  # TODO: make these configurable
            )
            self.text_decoder = get_peft_model(self.text_decoder, peft_config)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        forward_kwargs = {"output_hidden_states": True}
        layer_idx = -2  # TODO make this configurable
        modality_emb = self.modality_encoder(inputs, forward_kwargs)[self.modality_prediction_key][layer_idx][:, 1:, :]
        proj_modality_emb = self.modality_projector(modality_emb)

        if inputs[self.text_decoder.sample_key] is not None:
            text_emb = self.text_decoder.get_input_embeddings()(inputs[self.text_decoder.sample_key])
            # prepend projected modality embeddings to token embeddings
            input_emb = torch.cat((proj_modality_emb, text_emb), axis=1)
        else:
            input_emb = proj_modality_emb

        pos = torch.arange(0, input_emb.shape[1], dtype=torch.long, device=input_emb.device)
        pos = pos.unsqueeze(0)
        dec_inputs = {"input_ids": None}
        dec_inputs_kwargs = {"inputs_embeds": input_emb, "position_ids": pos}

        text_logits = self.text_decoder(dec_inputs, dec_inputs_kwargs)[self.text_decoder.prediction_key]
        return {self.prediction_key: text_logits}
