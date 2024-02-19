from abc import ABC, abstractmethod
from io import BytesIO
from typing import Generic, Optional, TypeVar

from PIL import Image
from transformers import PreTrainedTokenizer

T = TypeVar("T")


class Codec(ABC, Generic[T]):
    @abstractmethod
    def encode(self, obj: T) -> bytes:
        pass

    @staticmethod
    @abstractmethod
    def decode(serialized_obj: bytes) -> T:
        pass


class FixSizedCodec(Codec[T]):
    """Base class for fix-sized Codecs

    Fix-sized codecs are special in that they encode a sequence of values where
    each value is encoded by a fix number of bytes. The length of thegenerated
    bytestring is an integer multiple of `num_bytes_per_value`.
    """

    @classmethod
    @abstractmethod
    def num_bytes_per_value(cls) -> int:
        raise NotImplementedError


class HfTokenizerCodec(FixSizedCodec[str]):
    TOKEN_SIZE_IN_BYTES = 4

    @classmethod
    def num_bytes_per_value(cls) -> int:
        return cls.TOKEN_SIZE_IN_BYTES

    def __init__(
        self, tokenizer: PreTrainedTokenizer, max_length: Optional[int] = None, add_eos_token: bool = True
    ) -> None:
        # instantiate
        self.tokenizer = tokenizer
        self.add_eos_token = add_eos_token

        if add_eos_token:
            # get eos token in bytes to append to the end of each sequence
            eos_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)
            self.eos_token = eos_token.to_bytes(type(self).TOKEN_SIZE_IN_BYTES, byteorder="big")

        self.tokenizer_kwargs = (
            {} if max_length is None else dict(max_length=max_length - int(add_eos_token), truncation=True)
        )

    def encode(self, text: str) -> bytes:
        # tokenize text and convert the token ids to bytes
        tokens = [
            t.to_bytes(type(self).TOKEN_SIZE_IN_BYTES, byteorder="big")
            for t in self.tokenizer(text, **self.tokenizer_kwargs)["input_ids"]
        ]
        #
        if len(tokens) == 0:
            raise ValueError("Received empty sample")
        # add special eos token
        if self.add_eos_token:
            tokens.append(self.eos_token)

        # join byte strings
        return b"".join(tokens)

    @classmethod
    def decode(cls, serialized_tokens: bytes) -> str:
        return [
            int.from_bytes(serialized_tokens[i : i + cls.TOKEN_SIZE_IN_BYTES], byteorder="big")
            for i in range(0, len(serialized_tokens), cls.TOKEN_SIZE_IN_BYTES)
        ]


class PillowImageCodec(Codec[str]):
    def __init__(self, save_format: str = "png") -> None:
        self._format = save_format

    def encode(self, img_file_path: str) -> bytes:
        buf = BytesIO()
        # write image to buffer
        with Image.open(img_file_path) as img:
            img.save(buf, format=self._format)
        # retuen buffer content
        buf.seek(0)
        return buf.read()

    @staticmethod
    def decode(serialized_img: bytes) -> str:
        return Image.open(BytesIO(serialized_img))
