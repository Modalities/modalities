from abc import ABC, abstractmethod
from typing import List
from llm_gym.batch import DatasetBatch, InferenceResultBatch
from llm_gym.gpt2.gpt2_model import NNModel
import torch


class PredictPostProcessingIF(ABC):
    """
    Interface to perform prediction on the torch NN Model for the entire btach.
    """

    @abstractmethod
    def postprocess(self, result_batch: InferenceResultBatch) -> InferenceResultBatch:
        raise NotImplementedError


class PredictPostprocessingComponent:

    @staticmethod
    def post_process(result_batch: InferenceResultBatch, post_processors: List[PredictPostProcessingIF]) -> InferenceResultBatch:
        """
        Perform prediction on the torch NN Model for the entire btach.

        :params:
               result_batch (InferenceResultBatch): Prediction performed on the model.
               post_processors (List[PredictPostProcessingIF]): Batch number for which details to be logged.

        :returns:
            result_batch (InferenceResultBatch): Prediction performed on the model.
        """
        for post_processor in post_processors:
            result_batch = post_processor.postprocess(result_batch)
        return result_batch


class ModelInferenceComponent:
    def __init__(self, model: NNModel, post_processors: List[PredictPostProcessingIF] = None):
        self.model = model
        self.post_processors = post_processors if post_processors is not None else []

    def predict(self, batch: DatasetBatch, no_grad=True) -> InferenceResultBatch:
        """
        Perform prediction on the torch NN Model.

        :params:
               model (NNModel): Torch Neural Network module.
               batch (DatasetBatch): Train Dataset.
               post_processors (List[PredictPostProcessingIF]): Batch number for which details to be logged.

        :returns:
            result_batch (InferenceResultBatch): Prediction performed on the model.
        """
        if no_grad:
            with torch.no_grad():
                self.model.eval()  # TODO: check if excessive calling causes slow downs.  
                forward_result = self.model.forward(batch.samples)
        else:
            self.model.train()  # TODO: check if excessive calling causes slow downs. 
            forward_result = self.model.forward(batch.samples)

        result_batch = InferenceResultBatch(targets=batch.targets, predictions=forward_result)
        result_batch = PredictPostprocessingComponent.post_process(result_batch, post_processors=self.post_processors)
        return result_batch
