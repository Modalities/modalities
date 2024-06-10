from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import jq
import numpy as np
import webdataset as wds
from pydantic import BaseModel
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import IterableDataset
from torch.utils.data.dataset import Dataset as TorchdataSet
from tqdm import tqdm
from transformers import BatchEncoding

from modalities.config.config import PydanticTokenizerIFType
from modalities.tokenization.tokenizer_wrapper import TokenizerWrapper

from ..dataloader.large_file_lines_reader import LargeFileLinesReader
from .create_packed_data import EmbeddedStreamData


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


class ModalityEnum(Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class TransformConfig(BaseModel):
    pass


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


class TextTransformConfig(TransformConfig):
    tokenizer: TokenizerWrapper
    max_length: int = 77
    padding: str = "max_length"
    truncation: bool = True
    return_attention_mask: bool = True


class WebDatasetConfig(BaseModel):
    urls: Union[List[str], str]
    source_image_key: str
    image_key: str
    source_text_key: str
    text_key: str
    tokenizer: PydanticTokenizerIFType
    block_size: int
    num_samples: int
    image_transform_config: Optional[ImageTransformConfig] = None
    shardshuffle: Optional[int] = None
    repeat: bool = False
    resample: bool = False
    shuffle: int = 0


def dummy_nodesplitter(src, group=None):
    # This node splitter is not actually splitting the data over the nodes
    # but keeps the complete dataset on each node.
    # This is required so that each node has the same amount of data.
    # In the case of 25 shards and 16 ranks for example 7 ranks are
    # without data in the second iteration. This will cause a crash once all_gather is called.
    yield from src


class WebDataset(IterableDataset):
    def __init__(
        self,
        urls: Dict[Tuple[ModalityEnum, ...], Union[List[str], str]],
        modality_key_mapping: Dict[ModalityEnum, Tuple[str, str]],
        modality_transforms_configs: Dict[ModalityEnum, TransformConfig],
        num_samples: Dict[Tuple[ModalityEnum, ...], int],
        mixing_ratios: Optional[Dict[Tuple[ModalityEnum, ...], int]] = None,
        shardshuffle: int = 100,
        repeat: bool = False,
        resample: bool = True,
        shuffle_buffer: Optional[int] = 10_000,
    ):
        """WebDataset for loading and combining multimodal datasets.

        Args:
            urls: Webdataset urls for each modality combination.
                For example: {(ModalityEnum.IMAGE, ModalityEnum.TEXT): "/data/path/{00000..00012.tar"}}

            modality_key_mapping: Mapping from dataset keys to keys expected by the forward pass of the model.
                For example: {ModalityEnum.IMAGE: ("jpg", "image"), ModalityEnum.TEXT: ("text", "caption")}}
            modality_transforms_configs: The transform config for each modality.
            num_samples: The number of samples for each modality combination.
                For example: {(ModalityEnum.IMAGE, ModalityEnum.TEXT): 1_234_567}}
            mixing_ratios: Mixing ratios of the different modality combinations.
                For example: {
                    (ModalityEnum.IMAGE, ModalityEnum.TEXT): 0.7,
                    (ModalityEnum.VIDEO, ModalityEnum.TEXT): 0.3}
                }
            shardshuffle: Number of sharfs that should be used for shuffling. Defaults to 100.
            repeat: Repeat the dataset. Defaults to False.
            resample: Instead if iterating in order sample random shards.
                This has the issue that the model will see sample multiple times but if significantly more
                efficient. Defaults to True.
            shuffle_buffer: Number of samples that should be used for shuffling. Defaults to 10_000.
        """
        self.num_samples = num_samples
        self.total_num_samples = sum([self.num_samples[k] for k in urls.keys()])
        self.modality_key_mapping = modality_key_mapping
        self.modality_transforms_configs = modality_transforms_configs

        # Create webdatasets for each modality combination
        self.web_datasets = {
            k: wds.WebDataset(
                urls=u,
                nodesplitter=dummy_nodesplitter if not resample else None,
                shardshuffle=shardshuffle,
                repeat=repeat,
                handler=wds.ignore_and_continue,
                resampled=resample,
            )
            for k, u in urls.items()
        }

        # Setup mixing ratios
        self.mixing_ratios = mixing_ratios
        if self.mixing_ratios is None and len(self.web_datasets) > 1:
            uniform_ratio = 1 / len(self.web_datasets)
            self.mixing_ratios = {k: uniform_ratio for k in self.web_datasets.keys()}

        # Mapping between modality and the decode "function"
        self.modality_to_decode_fn = {
            ModalityEnum.TEXT: None,
            ModalityEnum.IMAGE: "pil",
            ModalityEnum.VIDEO: wds.torch_video,
            ModalityEnum.AUDIO: wds.torch_audio,
        }

        # Some transforms require objects such as image
        if ModalityEnum.IMAGE in self.modality_transforms_configs:
            self._timm_image_transform = create_transform(
                **self.modality_transforms_configs[ModalityEnum.IMAGE].model_dump()
            )

        # Mapping between modality and transform
        self.modality_to_transform_fn = {
            ModalityEnum.TEXT: self._transform_text,
            ModalityEnum.IMAGE: self._transform_image,
            ModalityEnum.VIDEO: self._transform_video,
            ModalityEnum.AUDIO: self._transform_audio,
        }

        for k, web_dataset in self.web_datasets:
            # Apply shuffling to samples
            if shuffle_buffer is not None and shuffle_buffer > 0:
                web_dataset.append(wds.filters.shuffle(shuffle_buffer))

            # Load the actual data
            for modality_key in k:
                transform_fn = self.modality_to_transform_fn[modality_key]
                self.append(wds.filters.map(transform_fn))

    def _transform_text(self, sample):
        source_key, target_key = self.modality_key_mapping[ModalityEnum.TEXT]
        config: TextTransformConfig = self.modality_transforms_configs[ModalityEnum.TEXT]
        batch_encoding: BatchEncoding = config.tokenizer.tokenizer(
            sample[source_key],
            max_length=config.block_size,
            padding=config.padding,
            truncation=config.truncation,
            return_attention_mask=config.return_attention_mask,
        )
        del sample[source_key]
        sample[target_key] = batch_encoding.input_ids
        sample["attention_mask"] = batch_encoding.attention_mask
        return sample

    def _transform_image(self, sample):
        source_key, target_key = self.modality_key_mapping[ModalityEnum.IMAGE]
        sample[target_key] = self._timm_image_transform(sample[source_key])
        del sample[source_key]
        return sample

    def _transform_video(self, sample):
        source_key, target_key = self.modality_key_mapping[ModalityEnum.VIDEO]
        # config: VideoTransformConfig = self.modality_transforms_configs[ModalityEnum.VIDEO]
        # TODO add video transform
        return sample

    def _transform_audio(self, sample):
        source_key, target_key = self.modality_key_mapping[ModalityEnum.AUDIO]
        # config: AudioTransformConfig = self.modality_transforms_configs[ModalityEnum.AUDIO]
        # TODO add audio transform
        return sample

    def __iter__(self):
        if len(self.web_datasets) > 1:
            datasets = []
            ratios = []
            for k in self.web_datasets.keys():
                datasets.append(self.web_datasets[k])
                ratios.append(self.mixing_ratios[k])
            dataset = wds.RandomMix(datasets, ratios)  # Apply mixing at sample level
            return iter(dataset)

        dataset = next(iter(self.web_datasets.values()))
        return iter(dataset)

    def __len__(self):
        return self.total_num_samples
