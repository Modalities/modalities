import torch.nn as nn
from torch.optim import AdamW


class OptimizerFactory:
    @staticmethod
    def get_adam_w(lr: float, model: nn.Module):
        model_parameters = model.parameters()
        optimizer = AdamW(params=model_parameters, lr=lr)
        return optimizer
