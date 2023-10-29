from functools import partial
from typing import Callable, Dict, List, Union
from llm_gym.exceptions import BatchStateError
import torch
from abc import ABC, abstractmethod


class TorchDeviceMixin(ABC):

    @staticmethod
    def _dict_tensor_to_device(d: Dict[str, Dict | torch.Tensor], device: torch.device) -> Dict[str, Dict | torch.Tensor]:
        partial_fun = partial(torch.Tensor.to, device=device)
        return TorchDeviceMixin.traverse_apply(ds=d, apply_fun=partial_fun)

    @staticmethod
    def _detach_dict_tensor(d: Dict[str, Dict | torch.Tensor]) -> Dict[str, Dict | torch.Tensor]:
        partial_fun = partial(torch.Tensor.detach)
        return TorchDeviceMixin.traverse_apply(ds=d, apply_fun=partial_fun)

    @staticmethod
    def traverse_apply(ds: Union[Dict, List, torch.Tensor],
                       apply_fun: Callable[[torch.Tensor], torch.Tensor]) -> Union[Dict, List, torch.Tensor]:
        if isinstance(ds, dict):
            return {k: TorchDeviceMixin.traverse_apply(d, apply_fun) for k, d in ds.items()}
        elif isinstance(ds, list):
            return [TorchDeviceMixin.traverse_apply(d, apply_fun) for d in ds]
        return apply_fun(ds)

    @property
    @abstractmethod
    def device(self) -> torch.device:
        raise NotImplementedError

    @abstractmethod
    def to(self, device: torch.device):
        raise NotImplementedError

    @abstractmethod
    def detach(self):
        raise NotImplementedError


class Batch(ABC):
    """Abstract class that defines the necessary methods any `Batch` implementation needs to implement.
    """
    pass


class DatasetBatch(Batch, TorchDeviceMixin):
    """A batch of samples and its targets. Used to batch train a model."""

    def __init__(self, samples: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], tags: torch.Tensor = None):
        self._samples = samples
        self._targets = targets

    @property
    def samples(self) -> Dict[str, torch.Tensor]:
        return self._samples

    @property
    def targets(self) -> Dict[str, torch.Tensor]:
        return self._targets

    def to(self, device: torch.device | int) -> "DatasetBatch":
        self._samples = {k: v.to(device) for k, v in self._samples.items()}
        self._targets = {k: v.to(device) for k, v in self._targets.items()}
        return self

    def detach(self):
        self._targets = {k: v.detach() for k, v in self._targets.items()}
        self._tags = self._tags.detach()
        self._samples = {k: v.detach() for k, v in self._samples.items()}

    @property
    def device(self) -> torch.device:
        key = list(self._samples.keys())[0]
        return self._samples[key].device

    def __len__(self) -> int:
        return len(self._samples)


class InferenceResultBatch(Batch, TorchDeviceMixin):
    """ Stores targets and predictions of an entire batch.
    """

    def __init__(self, targets: Dict[str, torch.Tensor] = None, predictions: Dict[str, torch.Tensor] = None):
        self._targets = targets if targets is not None else {}
        self._predictions = predictions if predictions is not None else {}
        self.to(self.device)

    def to_cpu(self):
        self.to(device=torch.device("cpu"))

    @property
    def device(self) -> torch.device:
        key = list(self._targets.keys())[0]
        return self._targets[key].device

    def to(self, device: torch.device):
        self._predictions = TorchDeviceMixin._dict_tensor_to_device(self._predictions, device)
        self._targets = TorchDeviceMixin._dict_tensor_to_device(self._targets, device)

    def detach(self):
        self._targets = TorchDeviceMixin._detach_dict_tensor(self._targets)
        self._predictions = TorchDeviceMixin._detach_dict_tensor(self._predictions)

    @property
    def predictions(self) -> Dict[str, torch.Tensor]:
        return self._predictions

    @property
    def targets(self) -> Dict[str, torch.Tensor]:
        return self._targets

    def get_predictions(self, key: str) -> torch.Tensor:
        if key not in self._predictions:
            raise BatchStateError(f"Key {key} not present in predictions!")
        return self._predictions[key]

    def get_targets(self, key: str) -> torch.Tensor:
        if key not in self._targets:
            raise BatchStateError(f"Key {key} not present in targets!")
        return self._targets[key]

    def __len__(self) -> int:
        return len(self.predictions)


class EvaluationResultBatch(Batch):
    """Data class for storing the results of a single or multiple batches. Also entire epoch results are stored in here.
    """

    def __init__(self, split_name: str, losses: Dict[str, torch.Tensor] = None, metrics: Dict[str, torch.Tensor] = None):
        self._losses = losses if losses is not None else {}
        self._metrics = metrics if metrics is not None else {}
        self._split_name = split_name

    @property
    def losses(self) -> Dict[str, torch.Tensor]:
        return self._losses

    @property
    def metrics(self) -> Dict[str, torch.Tensor]:
        return self._metrics

    @property
    def split_name(self) -> str:
        return self._split_name

    def __str__(self) -> str:
        eval_str = f"Evaluation result on dataset split ({self._dataset_split}):"
        eval_str += "\n\nlosses: " + "\n\t".join([f"{k}: {v}" for k, v in self._losses.mean().items()])
        eval_str += "\n\nmetrics: " + "\n\t".join([f"{k}: {v}" for k, v in self._metrics.mean().items()])
        eval_str += "\n==============================================="
        return eval_str
