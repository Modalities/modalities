import io
import tarfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torchaudio
import webdataset as wds
from pydantic import BaseModel

from modalities.__main__ import load_app_config_dict
from modalities.config.component_factory import ComponentFactory
from modalities.config.pydanctic_if_types import PydanticDataLoaderIFType
from modalities.registry.components import COMPONENTS
from modalities.registry.registry import Registry
from tests.conftest import _ROOT_DIR


def create_image_sample():
    img = np.random.randint(0, 255, size=(224, 224, 3)).astype(np.uint8)
    img = wds.writer.imageencoder(img, format="JPG")
    text = {"text0": "this is an image caption %d" % np.random.randint(10)}
    return img, text


@pytest.fixture(scope="session")
def image_tar_path(tmp_path_factory):
    data_path = str(tmp_path_factory.mktemp("data") / "images.tar")
    dataset_sink = wds.TarWriter(data_path)
    # 10 image samples
    for idx in range(10):
        img, text = create_image_sample()
        dataset_sink.write(
            {
                "__key__": "%02d" % idx,
                "jpg": img,
                "json": text,
            }
        )
    dataset_sink.close()
    return data_path


def create_audio_sample():
    sample_rate = 16000
    audio = torch.from_numpy(np.random.uniform(-1, 1, sample_rate)).unsqueeze(0)
    audio_buf = io.BytesIO()
    torchaudio.save(audio_buf, audio, sample_rate, format="wav")
    audio_buf.seek(0)
    text = "this is an audio caption %d" % np.random.randint(10)
    text_f = io.BytesIO()
    text_f.write(text.encode("utf-8"))
    text_f.seek(0)
    return audio_buf, text_f


@pytest.fixture(scope="session")
def audio_tar_path(tmp_path_factory):
    data_path = str(tmp_path_factory.mktemp("data") / "audio.tar")
    with tarfile.open(data_path, "w") as fp:
        # 25 audio samples
        for idx in range(25):
            key = "%02d" % idx
            wav, text = create_audio_sample()
            info = tarfile.TarInfo(key + ".wav")
            info.size = wav.getbuffer().nbytes
            fp.addfile(info, wav)
            info = tarfile.TarInfo(key + ".transcript.txt")
            info.size = text.getbuffer().nbytes
            fp.addfile(info, text)
    return data_path


@pytest.mark.parametrize(
    "mixing_ratios,resample,batch_size",
    [
        ([0.9, 0.1], False, 10),  # we run out of image samples after the second batch
        ([0.9, 0.1], True, 10),  # since we resample, there are enough samples for >2 batches
        ([0.7, 0.3], False, 20),  # the first batch won't have 0.7*20 samples
        ([0.3, 0.6], False, 10),  # ratios don't add up to 1
        ([0.8, 0.2], True, 100),
    ],
)
def test_web_dataloader(image_tar_path, audio_tar_path, mixing_ratios, resample, batch_size):
    class DataloaderTestModel(BaseModel):
        train_dataloader: PydanticDataLoaderIFType

    config_file_path = _ROOT_DIR / Path("tests/dataloader/yaml_configs/web_dataloader.yaml")
    config_dict = load_app_config_dict(config_file_path=config_file_path)
    config_dict["image_dataset"]["config"]["urls"] = image_tar_path
    config_dict["audio_dataset"]["config"]["urls"] = audio_tar_path
    config_dict["train_dataset"]["config"]["mixing_ratios"] = mixing_ratios
    config_dict["train_dataset"]["config"]["resample"] = resample
    config_dict["train_dataset"]["config"]["batch_size"] = batch_size
    config_dict["train_dataloader"]["config"]["batch_size"] = batch_size
    registry = Registry(COMPONENTS)
    component_factory = ComponentFactory(registry=registry)
    components = component_factory.build_components(config_dict=config_dict, components_model_type=DataloaderTestModel)

    expected_images = int(mixing_ratios[0] * batch_size)
    expected_audio = int(mixing_ratios[1] * batch_size)
    # if ratios don't add up to 1, extra samples are added to first modality
    remaining = batch_size - (expected_audio + expected_images)
    expected_images += remaining

    loader = iter(components.train_dataloader)

    # image, audio
    total_samples = [10, 25]
    seen_samples = [0, 0]

    for idx in range(5):
        batch_expected_images = expected_images
        batch_expected_audio = expected_audio
        try:
            batch = next(loader)
        except StopIteration:
            break

        if not resample:
            # if resample is False, the last batch may have less
            # samples than expected if one of the modalities
            # runs out of samples
            if total_samples[0] - seen_samples[0] < expected_images:
                expected_images - (total_samples[0] - seen_samples[0])
                batch_expected_images = total_samples[0] - seen_samples[0]
            if total_samples[1] - seen_samples[1] < expected_audio:
                expected_audio - (total_samples[1] - seen_samples[1])
                batch_expected_audio = total_samples[1] - seen_samples[1]

        assert batch.samples["images"].shape[0] == batch_expected_images
        seen_samples[0] += batch.samples["images"].shape[0]
        assert batch.samples["audio"].shape[0] == batch_expected_audio
        seen_samples[1] += batch.samples["audio"].shape[0]
        assert batch.samples["input_ids"].shape[0] == batch_expected_audio + batch_expected_images
        for idx in range(2):
            # reset if the complete dataset has been seen already
            if seen_samples[idx] == total_samples[idx]:
                seen_samples[idx] = 0
