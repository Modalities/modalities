from abc import ABC, abstractmethod

import torch.nn as nn


class RunningEnv(ABC, object):
    def __enter__(self) -> "RunningEnv":
        raise NotImplementedError

    def __exit__(self, type, value, traceback):
        raise NotImplementedError

    @abstractmethod
    def wrap_model(self, model: nn.Module, sync_module_states: bool) -> nn.Module:
        raise NotImplementedError
