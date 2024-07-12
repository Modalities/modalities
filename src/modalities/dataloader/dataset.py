from __future__ import annotations

import random
from enum import Enum
from pathlib import Path
from typing import Annotated, Dict, List, Optional, Tuple, Union

import jq
import numpy as np
import torch
import webdataset as wds
from pydantic import BaseModel, Field
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data.dataset import Dataset as TorchdataSet
from torchvision import transforms
from tqdm import tqdm
from transformers import BatchEncoding

from modalities.config.config import PydanticTokenizerIFType
from modalities.config.lookup_enum import LookupEnum
from modalities.config.pydanctic_if_types import PydanticThirdPartyTypeIF
from modalities.dataloader.create_packed_data import EmbeddedStreamData
from modalities.dataloader.large_file_lines_reader import LargeFileLinesReader
from modalities.tokenization.tokenizer_wrapper import TokenizerWrapper
from modalities.util import flatten_dict


class Dataset(TorchdataSet):
    def __init__(self, raw_data_path: Path, block_size: int, sample_key: str):
        self.raw_data_path = raw_data_path
        self.block_size = block_size
        self.sample_key = sample_key

    def _check_if_inbounds(self, idx: int):
        if not 0 <= idx < len(self):
            raise IndexError


class DummySampleDataType(str, Enum):
    FLOAT = "float"
    INT = "int"


class DummySampleConfig(BaseModel):
    sample_key: str
    sample_shape: Tuple[int, ...]
    sample_type: DummySampleDataType


class DummyDatasetConfig(BaseModel):
    num_samples: int
    sample_definition: List[DummySampleConfig]


class DummyDataset(Dataset):
    def __init__(self, num_samples: int, sample_definition: Tuple[DummySampleConfig]):
        """
        :param num_samples: Number of samples the dataset should generate.
        :param sample_definition: A list of tuples defining the dataset output.
            Each touple contains the sample key, shape and data type.
        """
        super().__init__(raw_data_path=None, block_size=None, sample_key=None)
        self.num_samples = num_samples
        self.sample_definition = sample_definition

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict:
        return self._create_random_sample()

    def _create_random_sample(self):
        sample = dict()
        for s in self.sample_definition:
            if s.sample_type == DummySampleDataType.FLOAT:
                data = np.random.randn(*s.sample_shape)
            elif s.sample_type == DummySampleDataType.INT:
                data = np.random.randint(low=0, high=512, size=s.sample_shape)
            else:
                raise NotImplementedError(f"DummyDataset does not support type { s.sample_type}")
            sample[s.sample_key] = data
        return sample


class MemMapDataset(Dataset):
    def __init__(
        self,
        raw_data_path: Path,
        block_size: int,
        tokenizer: TokenizerWrapper,
        sample_key: str,
        index_path: Optional[Path] = None,
        jq_pattern: str = ".text",
    ):
        """
        Pytorch Dataset with mmap support.

        :param raw_data_path: Path to a jsonl file, which holds text data
        :param block_size: alias for max sequence length. The amount of tokens the model can handle.
        :param tokenizer: PretrainedTokenizer required to tokenize text data on the fly.
        :param jq_pattern: jq-pattern applied on every jsonl-entry. Results are afterwards tokenized and packed
        :param index_path: Path to an index file, which indicates the start character/byte position
                           and length of samples given in `raw_data_path`.
                           If not defined, an index next to `raw_data_path` is picked,
                           by replacing its suffix with ".idx".
        :param sample_key: model-specific parameter to indicate where in the BatchEncoding the input_token_ids are.
                           TODO: If this setting should support multi-modal features using separately encoded inputs,
                            this needs to get replaced with a list of sample keys!
        """
        super().__init__(raw_data_path=raw_data_path, block_size=block_size, sample_key=sample_key)

        self.reader = LargeFileLinesReader(self.raw_data_path, index_path=index_path)
        self.jq_filter = jq.compile(jq_pattern)
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.reader)

    def __getitem__(self, idx: int) -> BatchEncoding:
        self._check_if_inbounds(idx)
        return self.tokenizer.tokenize(text=self.jq_filter.input_text(self.reader[idx]).first())


class PackedMemMapDatasetBase(Dataset):
    DATA_SECTION_LENGTH_IN_BYTES = EmbeddedStreamData.DATA_SECTION_LENGTH_IN_BYTES
    TOKEN_SIZE_DESCRIPTOR_LENGTH_IN_BYTES = EmbeddedStreamData.TOKEN_SIZE_DESCRIPTOR_LENGTH_IN_BYTES
    HEADER_SIZE_IN_BYTES = EmbeddedStreamData.HEADER_SIZE_IN_BYTES
    np_dtype_of_tokens_on_disk_from_bytes = {
        1: np.dtype(np.uint8).newbyteorder("<"),
        2: np.dtype(np.uint16).newbyteorder("<"),
        4: np.dtype(np.uint32).newbyteorder("<"),
    }
    type_converter_for_torch = {1: np.uint8, 2: np.int32, 4: np.int64}

    def __init__(self, raw_data_path: Path, block_size: int, sample_key: str):
        """
        Base class for packed memmapped datasets. The underlying dataset file has the structure:
        | header | data | index |
        The header contains information about the length of the subsequent data sequence and the amount of bytes
        required to represent tokens in the data section. The index contains the tuple information (start, end) in terms
         of byte positions.

        :param raw_data_path: Path to a packed binary file (*.pbin).
                              Use `modalities data pack_encoded_data` to create one based on a jsonl-file.
        :param block_size: alias for max sequence length. The amount of tokens the model can handle.
        :param sample_key: model-specific parameter to indicate where in the BatchEncoding the input_token_ids are.
                           TODO: If this setting should support multi-modal features using separately encoded inputs,
                            this needs to get replaced with a list of sample keys!
        """
        super().__init__(raw_data_path=raw_data_path, block_size=block_size, sample_key=sample_key)
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

    def _generate_packing_index(self) -> List[Tuple[int, int]]:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> BatchEncoding:
        self._check_if_inbounds(idx)
        offset, length = self._index[idx]
        tokens = np.frombuffer(
            self._embedded_stream_data.data, dtype=self._token_dtype_on_disk, count=length, offset=offset
        )
        # torch can't convert most uint-formats, therefore we infer regular int types
        tokens = tokens.astype(self._token_dtype_in_ram)
        return BatchEncoding(data={self.sample_key: tokens})


class PackedMemMapDatasetContinuous(PackedMemMapDatasetBase):
    def _generate_packing_index(self) -> List[Tuple[int, int]]:
        # get number of total tokens in file
        total_tokens = self._embedded_stream_data.data_len // self._token_size_in_bytes
        num_samples = total_tokens // self.block_size
        return [(i * self.block_size * self._token_size_in_bytes, self.block_size) for i in range(num_samples)]


class PackedMemMapDatasetMegatron(PackedMemMapDatasetBase):
    def _generate_packing_index(self) -> List[Tuple[int, int]]:
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
    input_size: Union[int, Tuple[int, int], Tuple[int, int, int]] = 224
    is_training: bool = False
    no_aug: bool = False
    train_crop_mode: Optional[str] = None
    scale: Optional[Tuple[float, float]] = None
    ratio: Optional[Tuple[float, float]] = None
    hflip: float = 0.5
    vflip: float = 0.0
    color_jitter: Union[float, Tuple[float, ...]] = 0.4
    color_jitter_prob: Optional[float] = None
    grayscale_prob: float = 0.0
    gaussian_blur_prob: float = 0.0
    auto_augment: Optional[str] = None
    interpolation: str = "bilinear"
    mean: Tuple[float, ...] = IMAGENET_DEFAULT_MEAN
    std: Tuple[float, ...] = IMAGENET_DEFAULT_STD
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


class RandomTemporalCrop:
    def __init__(self, num_frames):
        self.num_frames = num_frames

    def __call__(self, video):
        total_frames = len(video)
        start = random.randint(0, total_frames - self.num_frames)
        return video[start : start + self.num_frames].permute(0, 3, 1, 2)  # F C H W


class VideoTransformConfig(TransformConfig):
    input_size: Union[int, Tuple[int, int], Tuple[int, int, int]] = 224
    is_training: bool = False
    num_frames: int = 16


class VideoTransform(Transform):
    def __init__(
        self,
        input_size: Union[int, Tuple[int, int], Tuple[int, int, int]] = 224,
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


class MultimodalWebDatasetBuilderConfig(BaseModel):
    urls: Union[List[str], str]
    modality_key_mapping: Dict[ModalityEnum, Tuple[str, str]]
    modality_transforms: Dict[ModalityEnum, PydanticTransformIFType]
    num_samples: Annotated[int, Field(ge=1)]


# @register_component("dataset", "web_dataset_builder", MultimodalWebDatasetBuilderConfig)
class MultimodalWebDatasetBuilder:
    def __init__(
        self,
        urls: Union[List[str], str],
        modality_key_mapping: Dict[str, Tuple[str, str]],
        modality_transforms: Dict[str, Transform],
        num_samples: int,
    ):
        """A multimodal dataset instance for the WebDataset.

        Args:
            urls: A webdataset url. For example: "/data/path/{00000..00012.tar".
            modality_key_mapping: Mapping from dataset keys to keys expected by the forward pass of the model.
                For example: {ModalityEnum.IMAGE: ("jpg", "image"), ModalityEnum.TEXT: ("text", "caption")}}
            modality_transforms: The transforms for each modality.
            num_samples: The number of samples for each modality combination.
        """
        self.urls = urls
        self.modality_key_mapping = modality_key_mapping
        self.modality_transforms = modality_transforms
        assert self.modality_key_mapping.keys() == self.modality_transforms.keys()
        self.modalities = list(self.modality_key_mapping.keys())
        self.num_samples = num_samples
        self.web_dataset = None

        # Mapping between modality and the decode "function"
        self.modality_to_decode_fn = {
            ModalityEnum.TEXT: None,
            ModalityEnum.IMAGE: "pil",
            ModalityEnum.VIDEO: wds.torch_video,
            ModalityEnum.AUDIO: wds.torch_audio,
        }

        self.additional_extreacted_keys = []
        if ModalityEnum.TEXT in self.modality_transforms:
            self.additional_extreacted_keys.append("attention_mask")

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
        del sample[source_key]
        return sample

    def _transform_audio(self, sample):
        source_key, target_key = self.modality_key_mapping[ModalityEnum.AUDIO]
        # config: AudioTransformConfig = self.modality_transforms_configs[ModalityEnum.AUDIO]
        # TODO add audio transform
        sample[target_key] = sample[source_key]
        del sample[source_key]
        return sample

    def _flatten_sample(self, sample):
        return flatten_dict(sample)

    def _select_keys(self, sample):
        select_keys = self.additional_extreacted_keys + [v[1] for v in self.modality_key_mapping.values()]
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
    builders: List[PydanticMultimodalWebDatasetBuilderIFType]
    mixing_ratios: Optional[List[int]] = None
    shardshuffle: int = 100
    repeat: bool = False
    resample: bool = True
    shuffle_buffer: Optional[int] = 10_000


# @register_component("dataset", "web_dataset", MultimodalWebDatasetConfig)
class MultimodalWebDataset(wds.DataPipeline, wds.compat.FluidInterface):
    def __init__(
        self,
        builders: List[MultimodalWebDatasetBuilder],
        mixing_ratios: Optional[List[int]] = None,
        shardshuffle: int = 100,
        repeat: bool = False,
        resample: bool = True,
        shuffle_buffer: Optional[int] = 10_000,
    ):
        """WebDataset for loading and combining multimodal datasets.

        Args:
            builders: WebDatasetBuilder instances.
            mixing_ratios: Mixing ratios of the different modality combinations.
                For example: [0.3, 0.7]
            shardshuffle: Number of sharfs that should be used for shuffling. Defaults to 100.
            repeat: Repeat the dataset. Defaults to False.
            resample: Instead if iterating in order sample random shards.
                This has the issue that the model will see sample multiple times but if significantly more
                efficient. Defaults to True.
            shuffle_buffer: Number of samples that should be used for shuffling. Defaults to 10_000.
        """
        super().__init__()
        self.builders = builders
        assert len(builders) == 1, "Multiple dataset builders are not supported yet"  # TODO

        self.output_keys_by_modality = {}
        for b in builders:
            for k, v in b.modality_key_mapping.items():
                if k not in self.output_keys_by_modality:
                    self.output_keys_by_modality[k] = v[1]
                else:
                    assert (
                        self.output_keys_by_modality[k] == v[1]
                    ), "Output keys for the same modality of all builders should be the same."

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
            datasets = []
            for b in self.builders:
                datasets.append(b.web_dataset)
            dataset = wds.RandomMix(datasets, self.mixing_ratios)  # Apply mixing at sample level
            self.pipeline.append(dataset)
        else:
            self.pipeline.extend(self.builders[0].web_dataset.pipeline)

        self.with_length(sum([b.num_samples for b in self.builders]))
