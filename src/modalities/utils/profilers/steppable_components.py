import os

import torch
import torch.nn as nn

from modalities.batch import InferenceResultBatch
from modalities.loss_functions import Loss
from modalities.utils.profilers.batch_generator import DatasetBatchGeneratorIF
from modalities.utils.profilers.steppable_components_if import SteppableComponentIF


class SteppableForwardPass(SteppableComponentIF):
    """A steppable component that performs a forward pass on the model using batches from the dataset batch generator.
    Optionally computes the loss if a loss function is provided.
    The component is used for profiling.
    """

    def __init__(self, model: nn.Module, dataset_batch_generator: DatasetBatchGeneratorIF, loss_fn: Loss | None = None):
        """Initializes the SteppableForwardPass component.

        Args:
            model (nn.Module): The model to perform the forward pass on.
            dataset_batch_generator (DatasetBatchGeneratorIF): The dataset batch generator to provide input batches.
            loss_fn (Loss, optional): The loss function to compute the loss. Defaults to None.
        """
        self.model = model
        self.loss_fn = loss_fn
        self.dataset_batch_generator = dataset_batch_generator
        self.device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")

    def step(
        self,
    ) -> None:
        """Performs a single step of the forward pass and computes the loss if a loss function is provided."""
        batch = self.dataset_batch_generator.get_dataset_batch()
        batch.to(device=self.device)
        predictions = self.model(batch.samples)
        result_batch = InferenceResultBatch(targets=batch.targets, predictions=predictions)
        if self.loss_fn is not None:
            self.loss_fn(result_batch)
