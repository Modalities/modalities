import pytest
import torch

from modalities.models.coca.collator import CoCaCollatorFn

# shared config
N_EMBD = 768

# text_decoder_config
TEXT_DECODER_VOCAB_SIZE = 50_304
TEXT_DECODER_BLOCK_SIZE = 100

# vision_transformer_config
N_IMAGE_CLASSES = 1_000
IMG_SIZE = 224
N_IMG_CHANNELS = 3
N_FRAMES = 16

# audio_transformer_config
AUDIO_BLOCK_SIZE = 500
N_MELS = 128
SUB_SAMPLING_FACTOR = 4


def dummy_image_sample():
    input_image = torch.randn(N_IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
    input_text = torch.randint(0, TEXT_DECODER_VOCAB_SIZE, (TEXT_DECODER_BLOCK_SIZE,))
    attn_mask = torch.randint(0, 2, (TEXT_DECODER_BLOCK_SIZE,))
    return dict(
        images=input_image,
        input_ids=input_text,
        attention_mask=attn_mask,
    )


def dummy_video_sample():
    input_video = torch.randn(N_FRAMES, N_IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
    input_text = torch.randint(0, TEXT_DECODER_VOCAB_SIZE, (TEXT_DECODER_BLOCK_SIZE,))
    attn_mask = torch.randint(0, 2, (TEXT_DECODER_BLOCK_SIZE,))
    return dict(
        video=input_video,
        input_ids=input_text,
        attention_mask=attn_mask,
    )


def dummy_audio_sample():
    audio_features = torch.randn(AUDIO_BLOCK_SIZE * SUB_SAMPLING_FACTOR, N_MELS)
    audio_len = torch.tensor([N_IMAGE_CLASSES / SUB_SAMPLING_FACTOR]).type(torch.int16)
    input_text = torch.randint(0, TEXT_DECODER_VOCAB_SIZE, (TEXT_DECODER_BLOCK_SIZE,))
    attn_mask = torch.randint(0, 2, (TEXT_DECODER_BLOCK_SIZE,))
    return dict(
        audio=audio_features,
        audio_len=audio_len,
        input_ids=input_text,
        attention_mask=attn_mask,
    )


@pytest.mark.parametrize(
    "modality_sequence",
    [
        ("iiiii"),
        ("aaaaa"),
        ("vvvvv"),
        ("iiaav"),
        ("iaiav"),
        ("iviaa"),
        ("iaiavaivaiiiiaaaviaa"),
    ],
)
def test_collator(modality_sequence):
    sample_keys = ["input_ids"]
    target_keys = []
    text_sample_key = "input_ids"
    text_target_key = "target_ids"

    num_image = modality_sequence.count("i")
    num_audio = modality_sequence.count("a")
    num_video = modality_sequence.count("v")

    # sample_keys in the order: images, audio, video
    if num_image:
        sample_keys.append("images")
    if num_audio:
        sample_keys.append("audio")
        sample_keys.append("audio_len")
    if num_video:
        sample_keys.append("video")

    # create samples
    image_samples = []
    for idx in range(num_image):
        image_samples.append(dummy_image_sample())
    audio_samples = []
    for idx in range(num_audio):
        audio_samples.append(dummy_audio_sample())

    video_samples = []
    for idx in range(num_video):
        video_samples.append(dummy_video_sample())

    modality_samples = {"images": image_samples, "audio": audio_samples, "video": video_samples}

    collate_fn = CoCaCollatorFn(sample_keys, target_keys, text_sample_key, text_target_key)

    batch = []
    image_idx = 0
    video_idx = 0
    audio_idx = 0
    # create the batch according to the specified modality sequence
    for ch in modality_sequence:
        if ch == "i":
            batch.append(image_samples[image_idx])
            image_idx += 1
        if ch == "a":
            batch.append(audio_samples[audio_idx])
            audio_idx += 1
        if ch == "v":
            batch.append(video_samples[video_idx])
            video_idx += 1

    dataset_batch = collate_fn(batch)

    batch_idx = 0

    # regardless of the order of the modality sequence,
    # the batch (esp. input_ids and target_ids) should be in the same order as sample_keys
    # i.e. batch.samples['input_ids'] = [*image input_ids, *audio_input_ids, *video_input_ids]
    for modality_key in sample_keys:
        if modality_key in ["audio_len", "input_ids"]:
            continue
        if modality_key in dataset_batch.samples:
            for modality_idx, gt_sample in enumerate(modality_samples[modality_key]):
                assert torch.equal(gt_sample[modality_key], dataset_batch.samples[modality_key][modality_idx])
                assert torch.equal(gt_sample["input_ids"][:-1], dataset_batch.samples[text_sample_key][batch_idx])
                assert torch.equal(gt_sample["input_ids"][1:], dataset_batch.targets[text_target_key][batch_idx])
                assert torch.equal(gt_sample["attention_mask"][:-1], dataset_batch.samples["attention_mask"][batch_idx])
                assert torch.equal(gt_sample["attention_mask"][1:], dataset_batch.targets["attention_mask"][batch_idx])
                if modality_key == "audio":
                    assert torch.equal(gt_sample["audio_len"], dataset_batch.samples["audio_len"][modality_idx])
                batch_idx += 1
