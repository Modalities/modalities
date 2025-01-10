import os

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from pydantic import BaseModel

from modalities.config.lookup_enum import LookupEnum
from modalities.config.pydanctic_if_types import PydanticPytorchModuleType
from modalities.models.model import NNModel


class AnyMALMode(LookupEnum):
    MODALITY_ALIGNMENT_TRAINING = "modality_alignment"
    INSTRUCTION_TUNING_TRAINING = "instruction_tuning"
    INFERENCE = "inference"


class AnyMALConfig(BaseModel):
    """

    Args:
        text_decoder (`PydanticPytorchModuleType`):
            a trained LLM
        prediction_key (`str`):
            key for the output logits of the model
        image_encoder (`PydanticPytorchModuleType`), required if training an image-text model:
            a trained image encoder (such as VisionTransformer or CLIPVisionModel)
        image_projector_config (`PerceiverConfig`), required if training an image-text model:
            config for Perceiver, which projects embeddings from the image encoder
            to the text token embedding space
        audio_encoder (`PydanticPytorchModuleType`), required if training an audio-text model:
            a trained audio encoder such as AudioTransformer
        audio_projector (`PerceiverConfig`), required if training an audio-text model:
            config for Perceiver, which projects embeddings from the audio encoder
            to the text token embedding space
        video_encoder (`PydanticPytorchModuleType`), required if training a video-text model:
            a trained video encoder
        video_projector_config (`PerceiverConfig`), required if training an video-text model:
            config for Perceiver, which projects embeddings from the video encoder
            to the text token embedding space
        model_mode (`AnyMALMode`):
            training stage for the model. The first modality alignment stage trains the modality projector using
            captioned modality. The second instruction fine-tuning stage trains the modality projector
            and fine-tunes the LLM (using LoRA) using instructions, request and response.


    """

    text_decoder: PydanticPytorchModuleType
    prediction_key: str
    image_encoder: PydanticPytorchModuleType = None
    image_projector: PydanticPytorchModuleType = None
    audio_encoder: PydanticPytorchModuleType = None
    audio_projector: PydanticPytorchModuleType = None
    video_encoder: PydanticPytorchModuleType = None
    video_projector: PydanticPytorchModuleType = None
    lora_weights: str = None
    model_mode: AnyMALMode = AnyMALMode.MODALITY_ALIGNMENT_TRAINING


class AnyMAL(NNModel):
    """Implementation of the AnyMAL model for multimodal alignment and instruction tuning of LLMs.
    Based on the paper:AnyMAL: An Efficient and Scalable Any-Modality Augmented Language Model
    Link: https://arxiv.org/abs/2309.16058
    """

    def __init__(
        self,
        text_decoder: PydanticPytorchModuleType,
        prediction_key: str,
        image_encoder: PydanticPytorchModuleType = None,
        image_projector: PydanticPytorchModuleType = None,
        audio_encoder: PydanticPytorchModuleType = None,
        audio_projector: PydanticPytorchModuleType = None,
        video_encoder: PydanticPytorchModuleType = None,
        video_projector: PydanticPytorchModuleType = None,
        model_mode: AnyMALMode = AnyMALMode.MODALITY_ALIGNMENT_TRAINING,
        lora_weights: str = None,
        seed: int = None,
    ) -> None:
        weight_decay_groups = {
            "linear": [
                r"image_projector.*attention",
                r"image_projector.*fc[12]",
                r"image_projector.output_proj",
                r"image_projector.fc1",
                r"image_projector.fc2",
                r"lora_A",
                r"lora_B",
            ],
            "embedding": [r"image_projector.positional_embedding_fn"],
            "norm": [r"image_projector.*norm", r"image_projector.*norm_latents"],
            "parameter": [r"image_projector.latents"],
        }
        super().__init__(weight_decay_groups=weight_decay_groups, seed=seed)
        self.prediction_key = prediction_key

        if image_encoder is None and audio_encoder is None and video_encoder is None:
            raise ValueError("At least one modality encoder should be specified.")

        if image_encoder is not None:
            if image_projector is None:
                raise ValueError("Image projector should not be None.")
            self.image_encoder = image_encoder
            self.image_projector = image_projector
            for param in self.image_encoder.parameters():
                param.requires_grad = False
        elif audio_encoder is not None:
            if audio_projector is None:
                raise ValueError("Audio projector should not be None.")
            self.audio_encoder = audio_encoder
            self.audio_projector = audio_projector
            for param in self.audio_encoder.parameters():
                param.requires_grad = False
        elif video_encoder is not None:
            if video_projector is None:
                raise ValueError("Video projector should not be None.")
            self.video_encoder = video_encoder
            self.video_projector = video_projector
            for param in self.video_encoder.parameters():
                param.requires_grad = False

        self.text_decoder = text_decoder

        if model_mode == AnyMALMode.MODALITY_ALIGNMENT_TRAINING:
            for param in self.text_decoder.parameters():
                param.requires_grad = False
        elif model_mode == AnyMALMode.INSTRUCTION_TUNING_TRAINING:
            peft_config = LoraConfig(
                inference_mode=False,
                r=64,
                lora_alpha=128,
                lora_dropout=0.1,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],  # TODO: make these configurable
            )
            self.text_decoder = get_peft_model(self.text_decoder, peft_config)
            self.text_decoder.print_trainable_parameters()
        elif model_mode == AnyMALMode.INFERENCE and lora_weights is not None:
            self.text_decoder = PeftModel.from_pretrained(self.text_decoder, os.path.dirname(lora_weights))
            self.text_decoder = self.text_decoder.merge_and_unload()

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        num_image_samples = 0
        num_audio_samples = 0
        num_video_samples = 0
        total_samples = inputs["input_ids"].shape[0]
        if "images" in inputs:
            num_image_samples += inputs["images"].shape[0]
        if "audio" in inputs:
            num_audio_samples += inputs["audio"].shape[0]
        if "video" in inputs:
            num_video_samples += inputs["video"].shape[0]
        num_multimodal_samples = num_image_samples + num_audio_samples + num_video_samples
        num_unimodal_samples = total_samples - num_multimodal_samples

        if num_multimodal_samples == 0:
            text_logits = self.text_decoder(inputs)[self.text_decoder.prediction_key]
            return {self.prediction_key: text_logits}

        if num_image_samples:
            forward_kwargs = {"output_hidden_states": True}
            layer_idx = -2  # TODO make this configurable
            image_emb = self.image_encoder(inputs, forward_kwargs)[self.image_encoder.prediction_key][layer_idx][
                :, 1:, :
            ]
            proj_image_emb = self.image_projector(image_emb)
            if num_unimodal_samples:
                dummy_proj_image_emb = torch.zeros(
                    (num_unimodal_samples, proj_image_emb.shape[1], proj_image_emb.shape[2]),
                    dtype=proj_image_emb.dtype,
                    device=proj_image_emb.device,
                )
                proj_image_emb = torch.cat((proj_image_emb, dummy_proj_image_emb), axis=0)

        text_emb = self.text_decoder.get_input_embeddings()(inputs[self.text_decoder.sample_key])
        # prepend projected modality embeddings to token embeddings

        # <bos emb> <image emb> <text emb>
        input_emb = torch.cat((text_emb[:, :1], proj_image_emb, text_emb[:, 1:]), axis=1)

        pos = torch.arange(0, input_emb.shape[1], dtype=torch.long, device=input_emb.device)
        pos = pos.unsqueeze(0)
        dec_inputs = {"input_ids": None}
        dec_inputs_kwargs = {
            "inputs_embeds": input_emb,
            "position_ids": pos,
        }
        if "attention_mask" in inputs:
            dec_inputs_kwargs["attention_mask"] = inputs["attention_mask"]

        text_logits = self.text_decoder(dec_inputs, dec_inputs_kwargs)[self.text_decoder.prediction_key]
        return {self.prediction_key: text_logits}
