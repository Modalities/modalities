from dataclasses import field

import torch
from pydantic import BaseModel

from modalities.batch import DatasetBatch
from modalities.models.gpt2.collator import CollateFnIF


class CoCaCollateFnConfig(BaseModel):
    """
    Configuration class for CoCaCollateFn.

    Args:
        sample_keys (list[str]): List of samples keys.
        target_keys (list[str]): List of target keys.
        text_sample_key (str): Key for the text samples.
        text_target_key (str): Key for the text targets.
    """

    sample_keys: list[str]
    target_keys: list[str]
    text_sample_key: str
    text_target_key: str


class CoCaCollatorFn(CollateFnIF):
    """Collator function for CoCa model."""

    def __init__(self, sample_keys: list[str], target_keys: list[str], text_sample_key: str, text_target_key: str):
        """
        Initializes the CoCaCollatorFn object.

        Args:
            sample_keys (list[str]): List of samples keys.
            target_keys (list[str]): List of target keys.
            text_sample_key (str): Key for the text samples.
            text_target_key (str): Key for the text targets.

        Raises:
            ValueError: If `text_sample_key` is not part of `sample_keys`.
            ValueError: If `text_target_key` is part of `target_keys`.

        Returns:
            None
        """
        self.device: torch.device = field(default_factory=lambda: torch.device("cpu"))
        if text_sample_key not in sample_keys:
            raise ValueError(f"{text_sample_key} is not part of sample keys {sample_keys}")
        if text_target_key in target_keys:
            raise ValueError(
                f"{text_target_key} should not be part of target keys {target_keys}, "
                f"because {text_target_key} will generated based on {text_sample_key}"
            )
        self.sample_keys = sample_keys
        self.target_keys = target_keys
        self.text_sample_key = text_sample_key
        self.text_target_key = text_target_key

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> DatasetBatch:
        """
        Process a batch of data.

        Args:
            batch (list[dict[str, torch.Tensor]]): A list of dictionaries containing tensors
            representing the batch data.

        Returns:
            DatasetBatch: The processed batch data.

        Raises:
            None.
        """
        # only keys related to the other modalities (e.g. images, audio, video)
        modality_keys = [key for key in self.sample_keys if key not in ["audio_len", self.text_sample_key]]

        samples = {sample_key: [] for sample_key in self.sample_keys if sample_key != self.text_sample_key}
        text_samples = {sample_key: [] for sample_key in modality_keys}
        attention_masks = {sample_key: [] for sample_key in modality_keys}
        # gather samples by modality
        for sample in batch:
            text_sample_added = False  # make sure text is only added once per sample
            for sample_key in self.sample_keys:
                if sample_key in sample:
                    if sample_key in samples:
                        samples[sample_key].append(self._prepare_sample(sample[sample_key]))
                    if "attention_mask" in sample and sample_key in attention_masks and not text_sample_added:
                        attention_masks[sample_key].append(self._prepare_sample(sample["attention_mask"]))
                    if sample_key in text_samples and not text_sample_added:
                        text_samples[sample_key].append(self._prepare_sample(sample[self.text_sample_key]))
                        text_sample_added = True
        # remove keys with no samples
        for sample_key in modality_keys:
            if len(text_samples[sample_key]) == 0:
                del text_samples[sample_key]
            if len(attention_masks[sample_key]) == 0:
                del attention_masks[sample_key]
        # stack samples by modality
        for sample_key in self.sample_keys:
            if sample_key in samples:
                samples[sample_key] = torch.stack(samples[sample_key])
            if sample_key in text_samples:
                text_samples[sample_key] = torch.stack(text_samples[sample_key])
            if sample_key in attention_masks:
                attention_masks[sample_key] = torch.stack(attention_masks[sample_key])
        # stack input_ids and attention masks for all modalities
        samples[self.text_sample_key] = torch.cat([text_samples[sample_key] for sample_key in text_samples])
        samples["attention_mask"] = torch.cat([attention_masks[sample_key] for sample_key in attention_masks])

        ## TODO: this will not work when there is data from multiple datasets per batch
        targets = {
            target_key: torch.stack([self._prepare_sample(d[target_key]) for d in batch])
            for target_key in self.target_keys
        }

        # Create target for text input
        targets[self.text_target_key] = samples[self.text_sample_key][:, 1:].clone().detach()
        samples[self.text_sample_key] = samples[self.text_sample_key][:, :-1]

        if "attention_mask" in batch[0]:
            targets["attention_mask"] = samples["attention_mask"][:, 1:].clone().detach()
            samples["attention_mask"] = samples["attention_mask"][:, :-1]

        return DatasetBatch(targets=targets, samples=samples)

    @staticmethod
    def _prepare_sample(x):
        if isinstance(x, torch.Tensor):
            return x
        return torch.tensor(x)
