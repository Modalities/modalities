from abc import ABC, abstractmethod
from io import BytesIO
from typing import Generic, Optional, TypeVar

import numpy as np
import torch
import torchaudio
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
        # return buffer content
        buf.seek(0)
        return buf.read()

    @staticmethod
    def decode(serialized_img: bytes) -> str:
        return Image.open(BytesIO(serialized_img))


class TorchaudioAudioCodec(Codec[str]):
    N_FFT = 400
    HOP_LENGTH = 160

    def __init__(
        self,
        target_sample_rate: int = 16_000,
        n_mels: int = 80,
    ) -> None:
        self.target_sample_rate = target_sample_rate
        self.extract_mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_mels=n_mels,
            n_fft=type(self).N_FFT,
            hop_length=type(self).HOP_LENGTH,
        )

    def load_audio(
        self,
        audio_file_path: str,
    ) -> torch.Tensor:
        audio, sample_rate = torchaudio.load(
            audio_file_path,
        )

        return (
            audio.mean(dim=0),
            sample_rate,
        )

    def extract_log_mel_spectrogram(
        self,
        audio: torch.Tensor,
    ) -> torch.Tensor:
        ############################################
        # Feature extraction is quite similar to how it is done
        # for Radford, Alec, et al. "Robust speech recognition
        # via large-scale weak supervision." 2023 AKA Whisper.
        # Their code can be found here:
        # https://github.com/openai/whisper/blob/main/whisper/audio.py
        # MIT LICENSE: https://github.com/openai/whisper/blob/main/LICENSE
        ############################################

        mel_spec = self.extract_mel_spec(audio)
        log_mel_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_mel_spec = torch.maximum(log_mel_spec, log_mel_spec.max() - 8.0)
        log_mel_spec = (log_mel_spec + 4.0) / 4.0
        return log_mel_spec.transpose(1, 0)

    def resample(
        self,
        audio: torch.Tensor,
        sample_rate: int,
    ) -> torch.Tensor:
        resampler = torchaudio.transforms.Resample(
            sample_rate,
            self.target_sample_rate,
            dtype=audio.dtype,
        )
        return resampler(audio)

    def encode(
        self,
        audio_file_path: str,
    ) -> bytes:
        audio, sample_rate = self.load_audio(
            audio_file_path,
        )

        audio = (
            self.resample(
                audio,
                sample_rate,
            )
            if sample_rate != self.target_sample_rate
            else audio
        )

        log_mel_spec = self.extract_log_mel_spectrogram(
            audio,
        ).numpy()

        buf = BytesIO()
        np.save(buf, log_mel_spec)
        buf.seek(0)

        return buf.read()

    @staticmethod
    def decode(
        serialized_audio: bytes,
    ) -> np.ndarray:
        return np.load(BytesIO(serialized_audio))
