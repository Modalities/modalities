from typing import Annotated, Optional

import numpy as np
import torch
import torch.nn.functional as F
from einops import repeat
from pydantic import BaseModel, Field
from torch import nn

from modalities.models.audio_transformer.audio_transformer_model import AudioTransformer, AudioTransformerConfig
from modalities.models.coca.attention_pooling import AttentionPooling
from modalities.models.coca.multi_modal_decoder import MultiModalTextDecoder
from modalities.models.coca.text_decoder import TextDecoder
from modalities.models.model import ActivationType, NNModel
from modalities.models.vision_transformer.vision_transformer_model import VisionTransformer, VisionTransformerConfig
from modalities.nn.attention import AttentionConfig


class TextDecoderConfig(BaseModel):
    """
    Configuration class for the TextDecoder.


    Args:
        sample_key (str): The key for the samples.
        prediction_key (str): The key for the predictions.
        block_size (int): The block size. Must be greater than or equal to 1.
        vocab_size (int): The vocabulary size. Must be greater than or equal to 1.
        n_layer_text (int): The number of layers for processing text. Must be greater than or equal to 1.
        n_layer_multimodal_text (int): -. Must be greater than or equal to 1.
        n_head (int): The number of attention heads. Must be greater than or equal to 1.
        n_embd (int): The embedding size. Must be greater than or equal to 1.
        ffn_hidden (int): The hidden size for the feed-forward network. Must be greater than or equal to 1.
        dropout (float): The dropout rate. Must be greater than or equal to 0.0.
        bias (bool): Flag indicating whether to include bias in the model.
        attention_config (AttentionConfig): The attention configuration.
        activation (ActivationType): The activation type.
        epsilon (float): The epsilon value. Must be greater than or equal to 0.0.
    """

    sample_key: str
    prediction_key: str
    block_size: Annotated[int, Field(ge=1)]
    vocab_size: Annotated[int, Field(ge=1)]
    n_layer_text: Annotated[int, Field(ge=1)]
    n_layer_multimodal_text: Annotated[int, Field(ge=1)]
    n_head: Annotated[int, Field(ge=1)]
    n_embd: Annotated[int, Field(ge=1)]
    ffn_hidden: Annotated[int, Field(ge=1)]
    dropout: Annotated[float, Field(ge=0.0)]
    bias: bool
    attention_config: AttentionConfig
    activation: ActivationType
    epsilon: Annotated[float, Field(ge=0.0)]


class CoCaConfig(BaseModel):
    """
    Configuration class for CoCa model.

    Args:
        prediction_key (str): The key for the predictions.
        text_embd_prediction_key (str): The key for the text embeddings.
        logit_scale_prediction_key (str): The key for the logit scale
        text_cls_prediction_key (Optional[str]): The key for the text cls token.
        audio_embd_prediction_key (Optional[str]): The key for audio embeddings
        image_embd_prediction_key (Optional[str]): The key for image embeddings
        video_embd_prediction_key (Optional[str]): The key for video embeddings
        audio_cls_prediction_key (Optional[str]): Th key for the audio cls token
        audio_text_cls_prediction_key (Optional[str]): Th key for the text cls token associated with the audio samples
        image_cls_prediction_key (Optional[str]): Th key for the image cls token
        image_text_cls_prediction_key (Optional[str]): Th key for the text cls token associated with the image samples
        video_cls_prediction_key (Optional[str]): Th key for the video cls token
        video_text_cls_prediction_key (Optional[str]): Th key for the text cls token associated with the video samples
        modality_keys (list[str]): sample keys in the input associated with the input modalities
        individual_datasets (Optional[bool]): flag indicating whether
            there are separate datasets for different modalities
        is_audio_video (Optional[bool]): flag indicating whether the video samples contain audio
        audio_encoder_config (Optional[AudioTransformerConfig]): config for the audio encoder. Defaults to None.
        image_encoder_config (Optional[VisionTransformerConfig]): config  for the image encoder. Defaults to None
        video_encoder_config (Optional[VisionTransformerConfig]): config  for the video encoder. Defaults to None
        text_decoder_config (TextDecoderConfig): Configuration for the text decoder.
        n_pool_head (int): Number of attention heads for pooling.
        n_queries (int): Number of queries for attention pooling.
        bias_attn_pool (bool): Flag indicating whether to use bias in attention pooling.
        epsilon_attn_pool (float): Epsilon value for attention pooling.
        seed (Optional[int]): The random seed. Defaults to None

    """

    prediction_key: str = "logits"
    text_embd_prediction_key: str
    logit_scale_prediction_key: str
    text_cls_prediction_key: Optional[str] = None
    audio_embd_prediction_key: Optional[str] = None
    image_embd_prediction_key: Optional[str] = None
    video_embd_prediction_key: Optional[str] = None
    audio_cls_prediction_key: Optional[str] = None
    audio_text_cls_prediction_key: Optional[str] = None
    image_cls_prediction_key: Optional[str] = None
    image_text_cls_prediction_key: Optional[str] = None
    video_cls_prediction_key: Optional[str] = None
    video_text_cls_prediction_key: Optional[str] = None
    modality_keys: list[str]
    individual_datasets: Optional[bool] = False
    is_audio_video: Optional[bool] = False
    audio_encoder_config: Optional[AudioTransformerConfig] = None
    image_encoder_config: Optional[VisionTransformerConfig] = None
    video_encoder_config: Optional[VisionTransformerConfig] = None
    text_decoder_config: TextDecoderConfig
    n_pool_head: Annotated[int, Field(ge=1)]
    n_queries: Optional[Annotated[int, Field(ge=1)]]
    bias_attn_pool: bool
    epsilon_attn_pool: Annotated[float, Field(ge=0.0)]
    seed: Optional[int] = None


class CoCa(NNModel):
    """
    CoCa model

    The Contrastive Captioner (CoCa) is an encoder-decoder model that integrates the concepts of CLIP
    and generative models such as SimVLM by using contrastive and captioning losses for training.

    Paper: `CoCa: Contrastive Captioners are Image-Text Foundation Models`
    Link: https://arxiv.org/abs/2205.01917
    """

    def __init__(
        self,
        prediction_key: str,
        text_embd_prediction_key: str,
        logit_scale_prediction_key: str,
        text_cls_prediction_key: Optional[str],
        audio_embd_prediction_key: Optional[str],
        image_embd_prediction_key: Optional[str],
        video_embd_prediction_key: Optional[str],
        audio_cls_prediction_key: Optional[str],
        audio_text_cls_prediction_key: Optional[str],
        image_cls_prediction_key: Optional[str],
        image_text_cls_prediction_key: Optional[str],
        video_cls_prediction_key: Optional[str],
        video_text_cls_prediction_key: Optional[str],
        modality_keys: list[str],
        individual_datasets: Optional[bool],
        is_audio_video: Optional[bool],
        audio_encoder_config: Optional[AudioTransformerConfig],
        image_encoder_config: Optional[VisionTransformerConfig],
        video_encoder_config: Optional[VisionTransformerConfig],
        text_decoder_config: TextDecoderConfig,
        n_pool_head: int,
        n_queries: Optional[int],
        bias_attn_pool: bool,
        epsilon_attn_pool: float,
        seed: int = None,
    ) -> None:
        """
        Initializes the CocaModel object.

        Args:
            prediction_key (str): The key for the predictions.
            text_embd_prediction_key (str): The key for the text embeddings.
            logit_scale_prediction_key (str): The key for the logit scale
            text_cls_prediction_key (Optional[str]): The key for the text cls token.
            audio_embd_prediction_key (Optional[str]): The key for audio embeddings
            image_embd_prediction_key (Optional[str]): The key for image embeddings
            video_embd_prediction_key (Optional[str]): The key for video embeddings
            audio_cls_prediction_key (Optional[str]): Th key for the audio cls token
            audio_text_cls_prediction_key (Optional[str]): Th key for the text cls token
                associated with the audio samples
            image_cls_prediction_key (Optional[str]): Th key for the image cls token
            image_text_cls_prediction_key (Optional[str]): Th key for the text cls token
                associated with the image samples
            video_cls_prediction_key (Optional[str]): Th key for the video cls token
            video_text_cls_prediction_key (Optional[str]): Th key for the text cls token
                associated with the video samples
            modality_keys (list[str]): sample keys in the input associated with the input modalities
            individual_datasets (Optional[bool]): flag indicating whether there are separate datasets
                for different modalities
            is_audio_video (Optional[bool]): flag indicating whether the video samples contain audio
            audio_encoder_config (Optional[AudioTransformerConfig]): config for the audio encoder. Defaults to None.
            image_encoder_config (Optional[VisionTransformerConfig]): config  for the image encoder. Defaults to None
            video_encoder_config (Optional[VisionTransformerConfig]): config  for the video encoder. Defaults to None
            text_decoder_config (TextDecoderConfig): Configuration for the text decoder.
            n_pool_head (int): Number of attention heads for pooling.
            n_queries (int): Number of queries for attention pooling.
            bias_attn_pool (bool): Flag indicating whether to use bias in attention pooling.
            epsilon_attn_pool (float): Epsilon value for attention pooling.
            seed (Optional[int]): The random seed. Defaults to None

        Raises:
            ValueError: if none of the modality encoders are defined
            ValueError: if using individual dataset and none of the text cls tokens
                corresponding to the modalities is defined
            ValueError: if training on a single dataset and text_cls_prediction_key is not defined

        Returns:
            None
        """
        weight_decay_groups = {
            "linear": [r"attention", r"\.attn", r"\.cross_attn", r"\.post_subsampler", r"_ffmodule", r"mlp"],
            "conv": [r"embedding_fn\.conv", r"project", r"\.subsampler", r"pointwise_conv", r"depthwise_conv"],
            "embedding": [r"wte", r"wpe", r"positional_embedding", r"time_embd"],
            "norm": [r"norm", r"norm_latents", r"\.ln_", r"\.batch_norm", r"exit_ln"],
            "parameter": [r"_queries", r"logit_scale", r"\.latents", r"cls_token"],
        }
        super().__init__(weight_decay_groups=weight_decay_groups, seed=seed)

        if individual_datasets:
            if (
                not audio_text_cls_prediction_key
                and not image_text_cls_prediction_key
                and not video_text_cls_prediction_key
            ):
                raise ValueError("All text_cls_prediction_keys cannot be None")
        else:
            if not text_cls_prediction_key:
                raise ValueError("text_cls_prediction key cannot be None")
        if not audio_encoder_config and not image_encoder_config and not video_encoder_config:
            raise ValueError("Atleast one modality encoder config should be specified")

        self.prediction_key = prediction_key
        self.text_embd_prediction_key = text_embd_prediction_key
        self.logit_scale_prediction_key = logit_scale_prediction_key
        self.text_cls_prediction_key = text_cls_prediction_key

        self.audio_embd_prediction_key = audio_embd_prediction_key
        self.image_embd_prediction_key = image_embd_prediction_key
        self.video_embd_prediction_key = video_embd_prediction_key
        self.audio_cls_prediction_key = audio_cls_prediction_key
        self.audio_text_cls_prediction_key = audio_text_cls_prediction_key
        self.image_cls_prediction_key = image_cls_prediction_key
        self.image_text_cls_prediction_key = image_text_cls_prediction_key
        self.video_cls_prediction_key = video_cls_prediction_key
        self.video_text_cls_prediction_key = video_text_cls_prediction_key

        self.modality_keys = modality_keys
        self.individual_datasets = individual_datasets
        self.is_audio_video = is_audio_video

        self.n_pool_head = n_pool_head
        self.bias_attn_pool = bias_attn_pool
        self.epsilon_attn_pool = epsilon_attn_pool
        self.text_decoder_config = text_decoder_config

        self.image_sample_key = None
        if image_encoder_config is not None:
            self.image_sample_key = image_encoder_config.sample_key
            self.image_encoder, self.image_queries, self.image_attn_pool = self._init_modality(
                VisionTransformer,
                image_encoder_config,
                n_queries,
            )

        self.video_sample_key = None
        if video_encoder_config is not None:
            self.video_sample_key = video_encoder_config.sample_key
            self.video_encoder, self.video_queries, self.video_attn_pool = self._init_modality(
                VisionTransformer,
                video_encoder_config,
                n_queries,
            )

        self.audio_sample_key = None
        if audio_encoder_config is not None:
            self.audio_sample_key = audio_encoder_config.sample_key
            self.audio_encoder, self.audio_queries, self.audio_attn_pool = self._init_modality(
                AudioTransformer,
                audio_encoder_config,
                n_queries,
            )

        self.text_decoder = TextDecoder(
            sample_key=text_decoder_config.sample_key,
            prediction_key=text_embd_prediction_key,
            block_size=text_decoder_config.block_size + 1,  # +1 for the class token
            n_layer=text_decoder_config.n_layer_text,
            vocab_size=text_decoder_config.vocab_size,
            n_head=text_decoder_config.n_head,
            n_embd=text_decoder_config.n_embd,
            ffn_hidden=text_decoder_config.ffn_hidden,
            dropout=text_decoder_config.dropout,
            bias=text_decoder_config.bias,
            attention_config=text_decoder_config.attention_config,
            activation=text_decoder_config.activation,
            epsilon=text_decoder_config.epsilon,
        )
        self.multimodal_decoder = MultiModalTextDecoder(
            sample_key=text_embd_prediction_key,
            prediction_key=text_decoder_config.prediction_key,
            block_size=text_decoder_config.block_size,
            n_layer=text_decoder_config.n_layer_multimodal_text,
            vocab_size=text_decoder_config.vocab_size,
            n_head=text_decoder_config.n_head,
            n_embd=text_decoder_config.n_embd,
            ffn_hidden=text_decoder_config.ffn_hidden,
            is_audio_video=self.is_audio_video,
            dropout=text_decoder_config.dropout,
            bias=text_decoder_config.bias,
            attention_config=text_decoder_config.attention_config,
            activation=text_decoder_config.activation,
            epsilon=text_decoder_config.epsilon,
        )

        self.text_decoder.transformer.wte.weight = (
            self.multimodal_decoder.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying

        # Logit scale for contrastive loss
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def _init_modality(
        self, encoder_class: type, encoder_config: VisionTransformerConfig | AudioTransformerConfig, n_queries: int
    ) -> tuple[VisionTransformer | AudioTransformer, nn.Parameter, AttentionPooling]:
        # initialize modality encoder, returns a tuple containing the encoder, queries and attention pooling layer
        encoder = encoder_class(**dict(encoder_config))
        queries = nn.Parameter(torch.randn(n_queries + 1, encoder_config.n_embd))
        attn_pool = AttentionPooling(
            n_embd=encoder_config.n_embd,
            n_head=self.n_pool_head,
            bias=self.bias_attn_pool,
            epsilon=self.epsilon_attn_pool,
            attention_config=self.text_decoder_config.attention_config,
        )
        return encoder, queries, attn_pool

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the CoCa model.

        Args:
            inputs (dict[str, torch.Tensor]): Input dictionary containing the text and modality samples
            In case of multiple modalities, the 'input_ids' key contain the token ids for
            the text corresponding to all the modalities stacked together. Thus the length (batch size)
            of 'input_ids' will be equal to the sum of the lengths of the individual modality
            samples.

        Returns:
            dict[str, torch.Tensor]: Output dictionary containing
              - cls token(s) for the modality or modalities
              - text cls token(s) corresponding to the modality sample(s)
              - logits from the text decoder
              - logit_scale
        """
        output = {}

        # encode modalities
        image_embd = audio_embd = video_embd = None
        if self.image_sample_key:
            image_embd, image_cls_token = self._forward_encode_image(inputs)
            output[self.image_cls_prediction_key] = image_cls_token

        if self.audio_sample_key:
            audio_embd, audio_cls_token = self._forward_encode_audio(inputs)
            output[self.audio_cls_prediction_key] = audio_cls_token

        if self.video_sample_key:
            video_embd, video_cls_token = self._forward_encode_video(inputs)
            output[self.video_cls_prediction_key] = video_cls_token

        # encode text
        text_embd, text_cls_token = self._forward_encode_text(inputs)

        # decode modality + text
        if self.individual_datasets:  # multiple modalities (from different datasets)
            start = 0
            modality_logits = []
            # this ensures that we select the text input_ids corresponding to each modality_key in the order
            # they are stacked by the collator
            for modality_key in self.modality_keys:
                if modality_key == "images" and image_embd is not None:
                    image_text_cls_token = text_cls_token[start : start + len(image_embd)]
                    image_text_embd = text_embd[start : start + len(image_embd)]
                    image_logits = self._forward_decode(image_text_embd, image_embd)
                    output.update({self.image_text_cls_prediction_key: image_text_cls_token})
                    modality_logits.append(image_logits)
                    start = start + len(image_embd)
                if modality_key == "audio" and audio_embd is not None:
                    audio_text_cls_token = text_cls_token[start : start + len(audio_embd)]
                    audio_text_embd = text_embd[start : start + len(audio_embd)]
                    audio_logits = self._forward_decode(audio_text_embd, audio_embd)
                    output.update({self.audio_text_cls_prediction_key: audio_text_cls_token})
                    modality_logits.append(audio_logits)
                    start = start + len(audio_embd)
                if modality_key == "video" and video_embd is not None:
                    video_text_cls_token = text_cls_token[start : start + len(video_embd)]
                    video_text_embd = text_embd[start : start + len(video_embd)]
                    video_logits = self._forward_decode(video_text_embd, video_embd)
                    output.update({self.video_text_cls_prediction_key: video_text_cls_token})
                    modality_logits.append(video_logits)
                    start = start + len(video_embd)
            logits = torch.cat(modality_logits)
        elif audio_embd is not None and video_embd is not None:  # video dataset that contains audio
            modality_embd = {"audio": audio_embd, "video": video_embd}
            logits = self._forward_decode(text_embd, modality_embd)
            output.update({self.text_cls_prediction_key: text_cls_token})
        else:  # single modality
            output.update({self.text_cls_prediction_key: text_cls_token})
            if image_embd is not None:
                logits = self._forward_decode(text_embd, image_embd)
            elif audio_embd is not None:
                logits = self._forward_decode(text_embd, audio_embd)
            elif video_embd is not None:
                logits = self._forward_decode(text_embd, video_embd)

        output.update(
            {
                self.prediction_key: logits,
                self.logit_scale_prediction_key: self.logit_scale.exp(),
            }
        )
        return output

    def _forward_encode_image(self, inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # returns a tuple containing the image embeddings and cls token
        image_embd = self.image_encoder(inputs)[self.image_embd_prediction_key]
        queries = repeat(self.image_queries, "n d -> b n d", b=image_embd.shape[0])
        image_embd = self.image_attn_pool(queries, context=image_embd)
        image_embd, image_cls_token = image_embd[:, :-1, :], F.normalize(image_embd[:, -1, :], dim=-1)
        return image_embd, image_cls_token

    def _forward_encode_video(self, inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # returns a tuple containing the video embeddings and cls token
        video_embd = self.video_encoder(inputs)[self.video_embd_prediction_key]
        queries = repeat(self.video_queries, "n d -> b n d", b=video_embd.shape[0])
        video_embd = self.video_attn_pool(queries, context=video_embd)
        video_embd, video_cls_token = video_embd[:, :-1, :], F.normalize(video_embd[:, -1, :], dim=-1)
        return video_embd, video_cls_token

    def _forward_encode_audio(self, inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # returns a tuple containing the audio embeddings and cls token
        audio_embd = self.audio_encoder(inputs)[self.audio_embd_prediction_key]
        queries = repeat(self.audio_queries, "n d -> b n d", b=audio_embd.shape[0])
        audio_embd = self.audio_attn_pool(queries, context=audio_embd)
        audio_embd, audio_cls_token = audio_embd[:, :-1, :], F.normalize(audio_embd[:, -1, :], dim=-1)
        return audio_embd, audio_cls_token

    def _forward_encode_text(self, inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # returns a tuple containing the encoded text tensor and the cls token
        text_embd = self.text_decoder(inputs)[self.text_embd_prediction_key]
        text_embd, text_cls_token = text_embd[:, :-1, :], F.normalize(text_embd[:, -1, :], dim=-1)
        return text_embd, text_cls_token

    def _forward_decode(
        self, text_embd: torch.Tensor, modality_embd: list[torch.Tensor] | torch.Tensor
    ) -> torch.Tensor:
        # forward decode given the text and modality embedding(s)
        decoder_inputs = {
            self.text_embd_prediction_key: text_embd,
            "context": modality_embd,
        }
        decoder_outputs = self.multimodal_decoder(decoder_inputs)
        logits = decoder_outputs[self.multimodal_decoder.prediction_key]
        return logits
