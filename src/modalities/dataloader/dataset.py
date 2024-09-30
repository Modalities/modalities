from __future__ import annotations

import io
import random
import re
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import decord
import jq
import numpy as np
import torch
import torchaudio
import webdataset as wds
from pydantic import BaseModel, Field
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import IterableDataset
from torch.utils.data.dataset import Dataset as TorchdataSet
from torchvision.transforms import v2 as transforms
from tqdm import tqdm
from transformers import BatchEncoding

from modalities.config.config import PydanticTokenizerIFType
from modalities.config.lookup_enum import LookupEnum
from modalities.config.pydanctic_if_types import PydanticThirdPartyTypeIF
from modalities.dataloader.create_packed_data import EmbeddedStreamData
from modalities.dataloader.large_file_lines_reader import LargeFileLinesReader
from modalities.tokenization.tokenizer_wrapper import TokenizerWrapper
from modalities.util import flatten_dict

decord.bridge.set_bridge("torch")


class Dataset(TorchdataSet):
    """Dataset class."""

    def __init__(self, raw_data_path: Path, sample_key: str):
        """
        Initializes a Dataset object.

        Args:
            raw_data_path (Path): The path to the raw data.
            sample_key (str): The key used to access a sample in the dataset.
        """
        self.raw_data_path = raw_data_path
        self.sample_key = sample_key

    def _check_if_inbounds(self, idx: int):
        # check if the provided index is within the bounds of the dataset.
        if not 0 <= idx < len(self):
            raise IndexError


class DummySampleDataType(str, Enum):
    """
    DummySampleDataType is an enumeration class that represents the data types for dummy samples.

    Attributes:
        FLOAT (str): Represents the float data type.
        INT (str): Represents the int data type.
    """

    FLOAT = "float"
    INT = "int"
    CONSTANT = "const"


class DummySampleConfig(BaseModel):
    """
    DummySampleConfig class represents the configuration for a dummy sample.

    Attributes:
        sample_key (str): The key of the sample.
        sample_shape (tuple[int, ...]): The shape of the sample.
        sample_type (DummySampleDataType): The type of the sample.

    """

    sample_key: str
    sample_shape: tuple[int, ...]
    sample_type: DummySampleDataType


class DummyDatasetConfig(BaseModel):
    """
    DummyDatasetConfig is a configuration class for defining a dummy dataset.

    Attributes:
        num_samples (int): The number of samples in the dataset.
        sample_definition (list[DummySampleConfig]): The list of sample definitions in the dataset.
    """

    num_samples: int
    sample_definition: list[DummySampleConfig]


class DummyDataset(Dataset):
    """DummyDataset class."""

    def __init__(self, num_samples: int, sample_definition: tuple[DummySampleConfig]):
        """
        Initializes a DummyDataset object with the given number of samples and sample definition.
        When calling the __getitem__ method, the dataset will return a random sample based on the sample definition.

        Args:
            num_samples (int): The number of samples in the dataset.
            sample_definition (tuple[DummySampleConfig]): A list of tuples defining the dataset output.
                Each touple contains the sample key, shape and data type.

        Returns:
            None
        """
        super().__init__(raw_data_path=None, sample_key=None)
        self.num_samples = num_samples
        self.sample_definition = sample_definition

        self.VISION = 1

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves an item from the dataset at the specified index.

        Parameters:
            idx (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary representing the retrieved item.

        Note:
            idx is not used. Instedam the method returns a random sample.
        """
        return self._create_random_sample()

    def _create_random_sample(self) -> dict:
        # creates a random sample based on the sample definition
        sample = dict()
        for s in self.sample_definition:
            if s.sample_type == DummySampleDataType.FLOAT:
                data = np.random.randn(*s.sample_shape)
            elif s.sample_type == DummySampleDataType.INT:
                data = np.random.randint(low=0, high=512, size=s.sample_shape)
            elif s.sample_type == DummySampleDataType.CONSTANT:
                data = self.VISION
            else:
                raise NotImplementedError(f"DummyDataset does not support type { s.sample_type}")
            sample[s.sample_key] = data
        return sample


class MemMapDataset(Dataset):
    def __init__(
        self,
        raw_data_path: Path,
        tokenizer: TokenizerWrapper,
        sample_key: str,
        index_path: Optional[Path] = None,
        jq_pattern: str = ".text",
    ):
        """
        Initializes the MemMapDataset object that represents a PyTorch Dataset with mmap support.

        Args:
            raw_data_path (Path): Path to a JSONL file, which holds text data.
            tokenizer (TokenizerWrapper): The tokenizer object that is required to tokenize text data.
            sample_key (str): The key to access the sample in the BatchEncoding.
            index_path (Optional[Path], optional): The path to the index file which indicates
              the start character/byte position of documents. Defaults to None.
            jq_pattern (str, optional): The jq pattern to filter the data. Results are afterwards tokenized and packed.
              Defaults to ".text".

        Returns:
            None
        """
        super().__init__(raw_data_path=raw_data_path, sample_key=sample_key)

        self.reader = LargeFileLinesReader(self.raw_data_path, index_path=index_path)
        self.jq_filter = jq.compile(jq_pattern)
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.reader)

    def __getitem__(self, idx: int) -> BatchEncoding:
        """
        Retrieves the item at the given index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            BatchEncoding: The tokenized representation of the item.

        Raises:
            IndexError: If the index is out of bounds.
        """
        self._check_if_inbounds(idx)
        return self.tokenizer.tokenize(text=self.jq_filter.input_text(self.reader[idx]).first())


class PackedMemMapDatasetBase(Dataset):
    """PackedMemMapDatasetBase class."""

    DATA_SECTION_LENGTH_IN_BYTES = EmbeddedStreamData.DATA_SECTION_LENGTH_IN_BYTES
    TOKEN_SIZE_DESCRIPTOR_LENGTH_IN_BYTES = EmbeddedStreamData.TOKEN_SIZE_DESCRIPTOR_LENGTH_IN_BYTES
    HEADER_SIZE_IN_BYTES = EmbeddedStreamData.HEADER_SIZE_IN_BYTES
    np_dtype_of_tokens_on_disk_from_bytes = {
        1: np.dtype(np.uint8).newbyteorder("<"),
        2: np.dtype(np.uint16).newbyteorder("<"),
        4: np.dtype(np.uint32).newbyteorder("<"),
    }
    type_converter_for_torch = {1: np.uint8, 2: np.int32, 4: np.int64}

    def __init__(self, raw_data_path: Path, sample_key: str):
        """
        Initializes the PackedMemMapDatasetBase object.

        Args:
            raw_data_path (Path): Path to a packed binary file (*.pbin).
                Use `modalities data pack_encoded_data` to create one based on a JSONL-file.
            sample_key (str): The key to access the sample in the BatchEncoding.

        Raises:
            RuntimeError: If the token representation with the given size is not supported.

        Returns:
            None

        Note:
            TODO: sample_key should support multi-modal features using separately encoded inputs,
                  this needs to get replaced with a list of sample keys!
        """
        super().__init__(raw_data_path=raw_data_path, sample_key=sample_key)
        self._embedded_stream_data = EmbeddedStreamData(raw_data_path)
        self._token_size_in_bytes = self._embedded_stream_data.token_size_in_bytes
        try:
            self._token_dtype_on_disk = self.np_dtype_of_tokens_on_disk_from_bytes[self._token_size_in_bytes]
            self._token_dtype_in_ram = self.type_converter_for_torch[self._token_size_in_bytes]
        except KeyError:
            raise RuntimeError(
                f"Encountered a required token representation with {self._token_size_in_bytes},"
                " which is not supported. Consider using a smaller vocabulary."
            )
        self._index = self._generate_packing_index()

    def _generate_packing_index(self) -> list[tuple[int, int]]:
        # Generates the packing index for the dataset.
        # The index is list of tuples, where each tuple contains the offset and length in bytes.

        return self._embedded_stream_data.index_base

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self._index)

    def __getitem__(self, idx: int) -> BatchEncoding:
        """
        Retrieves the item at the given index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            BatchEncoding: The retrieved item as a BatchEncoding object.

        Raises:
            ValueError: If the length of the sample in bytes is not a multiple of `self._token_size_in_bytes`.
        """
        self._check_if_inbounds(idx)
        # offset and length in bytes
        offset_in_bytes, length_in_bytes = self._index[idx]
        if length_in_bytes % self._token_size_in_bytes != 0:
            raise ValueError(
                f"Length of the sample in bytes is not a multiple of {self._token_size_in_bytes}."
                f"Offset in bytes: {offset_in_bytes}, Length in bytes: {length_in_bytes}"
            )
        # numpy frombuffer takes the memmap object as the buffer
        # and indices the data section with the given offset (in bytes)
        # and length in indices of type self._token_dtype_on_disk
        num_tokens = length_in_bytes // self._token_size_in_bytes
        tokens = np.frombuffer(
            buffer=self._embedded_stream_data.data,
            dtype=self._token_dtype_on_disk,
            count=num_tokens,
            offset=offset_in_bytes,
        )
        # torch can't convert most uint-formats, therefore we infer regular int types
        tokens = tokens.astype(self._token_dtype_in_ram)
        return BatchEncoding(data={self.sample_key: tokens})


class PackedMemMapDatasetContinuous(PackedMemMapDatasetBase):
    """PackedMemMapDatasetContinuous class."""

    def __init__(self, raw_data_path: Path, sample_key: str, block_size: int):
        """
        Initializes the PackedMemMapDatasetContinuous object.

        Args:
            raw_data_path (Path): Path to a packed binary file (*.pbin).
                Use `modalities data pack_encoded_data` to create one based on a JSONL-file.
            sample_key (str): The key to access the sample in the BatchEncoding.
            block_size (int): The size of the block.

        Returns:
            None
        """
        self.block_size = block_size
        super().__init__(raw_data_path=raw_data_path, sample_key=sample_key)

    def _generate_packing_index(self) -> list[tuple[int, int]]:
        # Generates the packing index for the dataset.
        # A list of tuples representing the index, where each tuple contains the offset and length in bytes.

        # get number of total tokens in file
        total_tokens = self._embedded_stream_data.data_len // self._token_size_in_bytes
        if total_tokens < self.block_size:
            raise ValueError(
                f"Block size ({self.block_size}) is larger than the"
                "total number of tokens in the dataset ({total_tokens})."
            )
        if self.block_size < 2:
            raise ValueError("Block size must be at least 2.")
        # Given a fixed number of samples we can compute the total number of tokens as
        # num_tokens = block_size + (block_size-1) * (num_samples-1)
        # as the first sample always needs block_size many tokens and the following samples
        # each need block_size-1 many tokens (since we can reuse the last target token as the first input token
        # of the subsequent sample).
        num_samples = (total_tokens - self.block_size) // (self.block_size - 1) + 1
        # given num_samples we calculate the starting index and length of each sample as tuple.
        return [
            ((i * self.block_size - i) * self._token_size_in_bytes, self.block_size * self._token_size_in_bytes)
            for i in range(num_samples)
        ]


class PackedMemMapDatasetMegatron(PackedMemMapDatasetBase):
    def __init__(self, raw_data_path: Path, sample_key: str, block_size: int):
        self.block_size = block_size
        super().__init__(raw_data_path=raw_data_path, sample_key=sample_key)

    def _generate_packing_index(self) -> list[tuple[int, int]]:
        index = []
        curr_offset = self.HEADER_SIZE_IN_BYTES
        curr_len = 0
        block_size_in_bytes = self.block_size * self._token_size_in_bytes
        for segment_offset, segment_len in tqdm(self._embedded_stream_data.index_base):
            # When the sum of the length of the current previously seen samples doesn't
            # exceed block_size_in_bytes, we add the current segment length to the previous
            # ones and continue.
            if curr_len + segment_len < block_size_in_bytes:
                curr_len += segment_len
            # If the previous and current length equals block_size_in_bytes, we add the starting index
            # and the total sequences length to the index list as a new sample.
            elif curr_len + segment_len == block_size_in_bytes:
                index.append((curr_offset, self.block_size))
                curr_len = 0
                curr_offset += block_size_in_bytes
            # Else case is executed when the current and previous segment length exceed the block_size.
            # In this case we set the starting point of the next sample to the end of the current sample.
            # This way, the start of a sample is never in the middle of a sentence.
            else:
                index.append((curr_offset, self.block_size))
                if segment_len > block_size_in_bytes:
                    curr_offset += block_size_in_bytes
                    curr_len = 0
                else:
                    curr_offset = segment_offset
                    curr_len = segment_len
        return index


class ModalityEnum(LookupEnum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class TransformConfig(BaseModel):
    pass


class Transform:
    pass


PydanticTransformIFType = Annotated[Transform, PydanticThirdPartyTypeIF(Transform)]


class ImageTransformConfig(TransformConfig):
    input_size: int | tuple[int, int] | tuple[int, int, int] = 224
    is_training: bool = False
    no_aug: bool = False
    train_crop_mode: Optional[str] = None
    scale: Optional[tuple[float, float]] = None
    ratio: Optional[tuple[float, float]] = None
    hflip: float = 0.5
    vflip: float = 0.0
    color_jitter: float | tuple[float, ...] = 0.4
    color_jitter_prob: Optional[float] = None
    grayscale_prob: float = 0.0
    gaussian_blur_prob: float = 0.0
    auto_augment: Optional[str] = None
    interpolation: str = "bilinear"
    mean: tuple[float, ...] = IMAGENET_DEFAULT_MEAN
    std: tuple[float, ...] = IMAGENET_DEFAULT_STD
    re_prob: float = 0.0
    re_mode: str = "const"
    re_count: int = 1
    re_num_splits: int = 0
    crop_pct: Optional[float] = None
    crop_mode: Optional[str] = None
    crop_border_pixels: Optional[int] = None
    tf_preprocessing: bool = False
    use_prefetcher: bool = False
    separate: bool = False


# @register_component("transform", "image_transform", ImageTransformConfig)
class ImageTransform(Transform):
    def __init__(self, **kwargs):
        self._timm_image_transform = create_transform(**kwargs)

    def __call__(self, image):
        return self._timm_image_transform(image)


class TextTransformConfig(TransformConfig):
    tokenizer: PydanticTokenizerIFType
    max_length: int = 77
    padding: str = "max_length"
    truncation: bool = True
    return_attention_mask: bool = True


# @register_component("transform", "text_transform", TextTransformConfig)
class TextTransform(Transform):
    def __init__(
        self,
        tokenizer: TokenizerWrapper,
        max_length: int = 77,
        padding: str = "max_length",
        truncation: bool = True,
        return_attention_mask: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.return_attention_mask = return_attention_mask

    def __call__(self, text):
        batch_encoding: BatchEncoding = self.tokenizer.tokenizer(
            text,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_attention_mask=self.return_attention_mask,
        )
        return batch_encoding


class AudioTransformConfig(TransformConfig):
    """
    Configuration class for the audio transformation module.

    This class defines various parameters that control the behavior of the AudioTransform.
    These parameters include whether the module is in training mode, the number of mel-frequency bands,
    lengths for frequency and time domain masking during training, and the target block size for audio encoding.

    Attributes:
        is_training (bool): Whether the module is in training mode. Defaults to False.
        n_mels (int): Number of mel-frequency bands. Defaults to 128.
        freq_domain_mask_length (int): Length of frequency masking during training. Defaults to 30.
        time_domain_mask_length (int): Length of time masking during training. Defaults to 100.
        block_size_audio_encoder (int): The target block size for audio encoding.
    """

    is_training: bool = False
    n_mels: int = 128
    freq_domain_mask_length: int = 30
    time_domain_mask_length: int = 100
    block_size_audio_encoder: int


class AudioTransform(Transform):
    """
    An audio transformation module that processes raw audio into mel-spectrogram features.

    This module includes steps such as feature extraction, frequency and time domain masking during training,
    padding to match a fixed block size, and returns the processed features along with their length.
    """

    def __init__(
        self,
        block_size_audio_encoder: int,
        is_training: bool = False,
        n_mels: int = 128,
        freq_domain_mask_length: int = 30,
        time_domain_mask_length: int = 100,
    ):
        """
        Initializes the AudioTransform class.

        Args:
            block_size_audio_encoder (int): The target block size for audio encoding.
            is_training (bool, optional): Whether the module is in training mode. Defaults to False.
            n_mels (int, optional): Number of mel-frequency bands. Defaults to 128.
            freq_domain_mask_length (int, optional): Length of frequency masking. Defaults to 30.
            time_domain_mask_length (int, optional): Length of time masking. Defaults to 100.

        Returns:
            tuple[torch.Tensor, int]: A tuple containing the processed audio features and their length.
        """
        self.block_size_audio_encoder = block_size_audio_encoder
        self.is_training = is_training
        self.n_mels = n_mels
        self.freq_domain_mask_length = freq_domain_mask_length
        self.time_domain_mask_length = time_domain_mask_length

    def __call__(self, raw_audio: tuple[torch.Tensor, int]) -> tuple[torch.Tensor, int]:
        """
        Processes the input raw audio into mel-spectrogram features.

        Args:
            raw_audio (tuple[torch.Tensor, int]): A tuple containing the raw audio tensor and its sample rate.

        Returns:
            tuple[torch.Tensor, int]: A tuple containing the processed audio features and their length.
        """

        SUB_SAMPLING_FACTOR = 4  # reduce the number of features (i.e., time frames)

        self.extract_features = torchaudio.transforms.MelSpectrogram(n_mels=self.n_mels)

        if self.is_training:
            self.masking = torch.nn.Sequential(
                torchaudio.transforms.FrequencyMasking(freq_mask_param=self.freq_domain_mask_length),
                torchaudio.transforms.TimeMasking(time_mask_param=self.time_domain_mask_length),
            )

        log_mel_spec = torch.clamp(self.extract_features(raw_audio[0]), 1e-10).log10().squeeze(0)
        log_mel_spec = self.masking(log_mel_spec) if self.is_training else log_mel_spec
        feats_len = log_mel_spec.shape[-1] // SUB_SAMPLING_FACTOR

        assert feats_len * SUB_SAMPLING_FACTOR <= SUB_SAMPLING_FACTOR * self.block_size_audio_encoder
        log_mel_spec = torch.nn.functional.pad(
            log_mel_spec, (0, SUB_SAMPLING_FACTOR * self.block_size_audio_encoder - log_mel_spec.shape[-1])
        ).transpose(0, 1)
        return log_mel_spec, feats_len


class RandomTemporalCrop:
    def __init__(self, num_frames):
        self.num_frames = num_frames

    def __call__(self, video):
        total_frames = len(video)
        start = random.randint(0, total_frames - self.num_frames)
        return video[start : start + self.num_frames].permute(0, 3, 1, 2)  # F C H W


class VideoTransformConfig(TransformConfig):
    input_size: int | tuple[int, int] | tuple[int, int, int] = 224
    is_training: bool = False
    num_frames: int = 16


class VideoTransform(Transform):
    def __init__(
        self,
        input_size: int | tuple[int, int] | tuple[int, int, int] = 224,
        is_training: bool = False,
        num_frames: int = 16,
    ):
        self.spatial_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(input_size, antialias=True),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.temporal_transform = RandomTemporalCrop(num_frames=num_frames)

    def __call__(self, video):
        video = video[0]
        video = self.temporal_transform(video)
        return self.spatial_transform(video)


def decord_video(key: str, data: bytes) -> None | tuple[torch.Tensor, Optional[torch.Tensor], int]:
    """
    Based on the torch_video decoder in webdataset
    https://github.com/webdataset/webdataset/blob/main/webdataset/autodecode.py#L394

    Decode a video file using Decord and optionally extract audio.

    This function decodes a video file from the provided data.
    It first checks if the file extension is one of the supported formats.
    If an audio stream exists, it extracts the audio with a mean across channels (if there are multiple).
    It then uses Decord to decode uniformly sampled frames from the video.

    Parameters:
        key (str): The key or identifier for the video data.
        data (bytes): The binary data of the video file.

    Returns:
        tuple: A tuple containing the decoded video frames, audio tensor (if available), and audio sample rate.
            If no audio stream exists, the audio tensor will be None.
    """
    extension = re.sub(r".*[.]", "", key)
    if extension not in "mp4 ogv mjpeg avi mov h264 mpg webm wmv".split():
        return None

    audio = None
    audio_sample_rate = -1
    stream = torchaudio.io.StreamReader(data)
    for idx in range(stream.num_src_streams):
        if stream.get_src_stream_info(idx).media_type == "audio":
            audio, audio_sample_rate = torchaudio.load(data)
            if audio.shape[0] > 1:  # more than one audio channel
                audio = torch.mean(audio, dim=0, keepdim=True)
            break

    file_obj = io.BytesIO(data)
    vr = decord.VideoReader(file_obj)
    clip_num_frames = 64
    # sample clip_num_frames uniformly from the full video
    frame_ids = torch.linspace(0, len(vr) - 1, clip_num_frames, dtype=torch.int64)
    frames = vr.get_batch(frame_ids.tolist())  # T x H x W x C

    return (frames, audio, audio_sample_rate)


def torch_audio(key: str, data: bytes) -> None | tuple[torch.Tensor, int]:
    """
    Based on the torch_audio decoder in webdataset
    https://github.com/webdataset/webdataset/blob/main/webdataset/autodecode.py#L418

    Decode an audio file using torchaudio.

    This function decodes an audio file from the provided data.
    It first checks if the file extension is one of the supported formats.
    If there are multiple channels in the audio file, it averages them to produce a mono audio tensor.

    Parameters:
        key (str): The key or identifier for the audio data.
        data (bytes): The binary data of the audio file.

    Returns:
        tuple: A tuple containing the decoded audio tensor and its sample rate. If the file extension is not supported,
               the function will return None.
    """

    extension = re.sub(r".*[.]", "", key)
    valid_extensions = "mp4 ogv mjpeg avi mov h264 mpg webm wmv flac mp3 sox wav m4a ogg wma".split()
    if extension not in valid_extensions:
        return None

    audio, sample_rate = torchaudio.load(data)
    if audio.shape[0] > 1:  # more than one channel
        audio = torch.mean(audio, dim=0, keepdim=True)
    return (audio, sample_rate)


def fixed_ratio_round_robin(*sources, samples_per_batch):
    sources = list(sources)
    remaining_samples_in_batch = samples_per_batch.copy()
    i = 0
    while len(sources) > 0:
        try:
            sample = next(sources[i])
            remaining_samples_in_batch[i] -= 1

            # reset
            if sum(remaining_samples_in_batch) == 0:
                remaining_samples_in_batch = samples_per_batch.copy()

            # go to next source which has some remaining samples
            i = (i + 1) % len(sources)
            while remaining_samples_in_batch[i] == 0:
                i = (i + 1) % len(sources)
            yield sample
        except StopIteration:
            del sources[i]


class FixedRatioRoundRobinMix(IterableDataset):
    def __init__(
        self,
        datasets: list[wds.WebDataset],
        mixing_ratios: list[float],
        batch_size: int,
    ):
        """An iterator for a list of datasets.
        Samples are yielded in a round robin manner
        with a fixed ratio of samples per dataset. There is no random sampling, so the number of
        samples per modality is guaranteed to be fixed per batch.

        Args:
            datasets (list[WebDataset]): a list of WebDatasets to be iterated over
            mixing_ratios (list[float]): the ratio of samples from each dataset that should be present in a batch
            batch_size (int): size of batch containing samples from all datasets in the specified ratio
        """
        self.datasets = datasets
        self.samples_per_batch = [int(batch_size * ratio) for ratio in mixing_ratios]
        # ensure ratio sums up to 1.0
        self.samples_per_batch[0] += batch_size - sum(self.samples_per_batch)

    def __iter__(self):
        """
        Returns:
            an iterator over the source datasets
        """
        sources = [iter(d) for d in self.datasets]
        return fixed_ratio_round_robin(*sources, samples_per_batch=self.samples_per_batch)


class MultimodalWebDatasetBuilderConfig(BaseModel):
    urls: list[str] | str
    modality_key_mapping: dict[ModalityEnum, tuple[str, str]]
    modality_transforms: dict[ModalityEnum, PydanticTransformIFType]
    is_audio_video: Optional[bool] = False
    num_samples: Annotated[int, Field(ge=1)]


# @register_component("dataset", "web_dataset_builder", MultimodalWebDatasetBuilderConfig)
class MultimodalWebDatasetBuilder:
    def __init__(
        self,
        urls: list[str] | str,
        modality_key_mapping: dict[str, tuple[str, str]],
        modality_transforms: dict[str, Transform],
        is_audio_video: bool,
        num_samples: int,
    ):
        """A multimodal dataset instance for the WebDataset.

        Args:
            urls (list[str] or str): A webdataset url. For example: "/data/path/{00000..00012}.tar".
            modality_key_mapping (dict[str, tuple[str, str]]): Mapping from dataset keys to keys
                expected by the forward pass of the model.
                For example: {ModalityEnum.IMAGE: ("jpg", "image"), ModalityEnum.TEXT: ("text", "caption")}}
            modality_transforms (dict[str, Transform]): The transforms for each modality as a dictionary.
            is_audio_video (bool): Whether the dataset is a video dataset which contains audio
            num_samples (int): The number of samples for each modality combination.

        Returns:
            None
        """
        self.urls = urls
        self.is_audio_video = is_audio_video
        self.modality_key_mapping = modality_key_mapping
        self.modality_transforms = modality_transforms
        # transforms should be specified for all modality_key mappings,
        # but we can also specify more transforms than necessary
        # so modality_key_mappings should be a subset of modality_transforms
        assert set(self.modality_key_mapping.keys()).issubset(self.modality_transforms.keys())
        self.modalities = list(self.modality_key_mapping.keys())
        self.num_samples = num_samples
        self.web_dataset = None

        # Mapping between modality and the decode "function"
        self.modality_to_decode_fn = {
            ModalityEnum.TEXT: None,
            ModalityEnum.IMAGE: "pil",
            ModalityEnum.VIDEO: decord_video,
            ModalityEnum.AUDIO: wds.torch_audio,
        }

        self.additional_extracted_keys = []
        if ModalityEnum.TEXT in self.modality_transforms:
            self.additional_extracted_keys.append("attention_mask")

        if ModalityEnum.AUDIO in self.modality_transforms or ModalityEnum.VIDEO in self.modality_transforms:
            self.additional_extracted_keys.append("audio_len")

        # Mapping between modality and transform
        self.modality_to_transform_fn = {
            ModalityEnum.TEXT: self._transform_text,
            ModalityEnum.IMAGE: self._transform_image,
            ModalityEnum.VIDEO: self._transform_video,
            ModalityEnum.AUDIO: self._transform_audio,
        }

    def prepare(
        self, shardshuffle: int = 100, resample: bool = True, repeat: bool = False, shuffle_buffer: int = 10_000
    ):
        """
        Prepares a WebDataset object as a pipeline that includes shuffling, decoding data, and transformations

        Args:
            shardshuffle (int): Number of shards that should be used for shuffling. Defaults to 100.
            resample (bool): Instead of iterating in order sample random shards.
                This has the issue that the model will see sample multiple times but is significantly more
                efficient. Defaults to True.
            repeat (bool): Repeat the dataset. Defaults to False.
            shuffle_buffer (Optional[int]): Number of samples that should be used for shuffling. Defaults to 10_000.

        Returns:
            None

        """
        self.web_dataset = wds.WebDataset(
            urls=self.urls,
            nodesplitter=self.dummy_nodesplitter if not resample else None,
            shardshuffle=shardshuffle,
            repeat=repeat,
            handler=wds.ignore_and_continue,
            resampled=resample,
        )

        # Apply shuffling to samples
        if shuffle_buffer is not None and shuffle_buffer > 0:
            self.web_dataset.append(wds.filters.shuffle(shuffle_buffer))

        # Flatten the json structure for convenience
        self.web_dataset.append(wds.filters.decode(partial=True))  # Decode json byte string
        self.web_dataset.append(wds.filters.map(self._flatten_sample))

        # Load the actual data
        for modality_key in self.modalities:
            decode_fn = self.modality_to_decode_fn[modality_key]
            if decode_fn is None:
                continue
            self.web_dataset.append(wds.filters.decode(decode_fn, partial=True))

        # Transform the data
        for modality_key in self.modalities:
            transform_fn = self.modality_to_transform_fn[modality_key]
            self.web_dataset.append(wds.filters.map(transform_fn))

        self.web_dataset.append(wds.filters.map(self._select_keys))

    def _transform_text(self, sample):
        source_key, target_key = self.modality_key_mapping[ModalityEnum.TEXT]
        transform: TextTransform = self.modality_transforms[ModalityEnum.TEXT]
        batch_encoding: BatchEncoding = transform(sample[source_key])
        del sample[source_key]
        sample[target_key] = batch_encoding.input_ids
        sample["attention_mask"] = batch_encoding.attention_mask
        return sample

    def _transform_image(self, sample):
        source_key, target_key = self.modality_key_mapping[ModalityEnum.IMAGE]
        transform: TextTransform = self.modality_transforms[ModalityEnum.IMAGE]
        sample[target_key] = transform(sample[source_key])
        del sample[source_key]
        return sample

    def _transform_video(self, sample):
        source_key, target_key = self.modality_key_mapping[ModalityEnum.VIDEO]
        transform: VideoTransform = self.modality_transforms[ModalityEnum.VIDEO]
        sample[target_key] = transform(sample[source_key])
        # if the video contains audio
        if sample[source_key][1] is not None and ModalityEnum.AUDIO in self.modality_transforms and self.is_audio_video:
            transform: AudioTransform = self.modality_transforms[ModalityEnum.AUDIO]
            sample["audio"], sample["audio_len"] = transform((sample[source_key][1], sample[source_key][2]))
            if "audio" not in self.additional_extracted_keys:
                self.additional_extracted_keys.append("audio")
        del sample[source_key]
        return sample

    def _transform_audio(self, sample: dict):
        # Apply audio transforms to the input sample.
        source_key, target_key = self.modality_key_mapping[ModalityEnum.AUDIO]
        transform: AudioTransform = self.modality_transforms[ModalityEnum.AUDIO]
        sample[target_key], sample["audio_len"] = transform(sample[source_key])
        del sample[source_key]
        return sample

    def _flatten_sample(self, sample):
        return flatten_dict(sample)

    def _select_keys(self, sample):
        # only select the required keys from the sample
        # i.e. the keys specified in modality_key_mapping
        # and the additional_extracted_keys
        select_keys = self.additional_extracted_keys + [v[1] for v in self.modality_key_mapping.values()]
        new_sample = {}
        for k, v in sample.items():
            if k not in select_keys:
                continue
            new_sample[k] = v
        return new_sample

    @staticmethod
    def dummy_nodesplitter(src, group=None):
        # This node splitter is not actually splitting the data over the nodes
        # but keeps the complete dataset on each node.
        # This is required so that each node has the same amount of data.
        # In the case of 25 shards and 16 ranks for example 7 ranks are
        # without data in the second iteration. This will cause a crash once all_gather is called.
        # This is only relevant for validation.
        yield from src


PydanticMultimodalWebDatasetBuilderIFType = Annotated[
    MultimodalWebDatasetBuilder, PydanticThirdPartyTypeIF(MultimodalWebDatasetBuilder)
]


class MultimodalWebDatasetConfig(BaseModel):
    builders: list[PydanticMultimodalWebDatasetBuilderIFType]
    batch_size: Optional[int] = None
    mixing_ratios: Optional[list[float]] = None
    shardshuffle: int = 100
    repeat: bool = False
    resample: bool = True
    shuffle_buffer: Optional[int] = 10_000


# @register_component("dataset", "web_dataset", MultimodalWebDatasetConfig)
class MultimodalWebDataset(wds.DataPipeline, wds.compat.FluidInterface):
    def __init__(
        self,
        builders: list[MultimodalWebDatasetBuilder],
        batch_size: int = None,
        mixing_ratios: Optional[list[float]] = None,
        shardshuffle: int = 100,
        repeat: bool = False,
        resample: bool = True,
        shuffle_buffer: Optional[int] = 10_000,
    ):
        """WebDataset for loading and combining multimodal datasets.

        Args:
            builders: WebDatasetBuilder instances.
            batch_size (int): batch size per device
            mixing_ratios (Optinal[list[float]]): Mixing ratios of the different modality combinations.
                For example: [0.3, 0.7]
            shardshuffle (int): Number of shards that should be used for shuffling. Defaults to 100.
            repeat (bool): Repeat the dataset. Defaults to False.
            resample (bool): Instead of iterating in order sample random shards.
                This has the issue that the model will see sample multiple times but is significantly more
                efficient. Defaults to True.
            shuffle_buffer (Optional[int]): Number of samples that should be used for shuffling. Defaults to 10_000.

        Raises:
            NotImplementedError: if multiple builders are specified and at least one builder contains a
                    video dataset which contains audio
            ValueError: if multiple builders are specified and batch size is None

        Returns:
            None
        """
        super().__init__()
        self.builders = builders

        for builder in self.builders:
            if builder.is_audio_video and len(self.builders) > 1:
                raise NotImplementedError(
                    "It is not yet possible to include a video-audio dataset with other types of modalities"
                )

        # Build datasets
        [
            b.prepare(shardshuffle=shardshuffle, resample=resample, repeat=repeat, shuffle_buffer=shuffle_buffer)
            for b in self.builders
        ]

        # Setup mixing ratios
        self.mixing_ratios = mixing_ratios
        if self.mixing_ratios is None:
            uniform_ratio = 1 / len(self.builders)
            self.mixing_ratios = [uniform_ratio for _ in self.builders]
        assert len(self.mixing_ratios) == len(self.builders)

        if len(self.builders) > 1:
            if batch_size is None:
                raise ValueError("batch_size cannot be None if multiple builders are used")
            datasets = []
            for b in self.builders:
                datasets.append(b.web_dataset)
            dataset = FixedRatioRoundRobinMix(datasets, self.mixing_ratios, batch_size)  # Apply mixing at sample level
            self.pipeline.append(dataset)
        else:
            self.pipeline.extend(self.builders[0].web_dataset.pipeline)

        self.with_length(sum([b.num_samples for b in self.builders]))
