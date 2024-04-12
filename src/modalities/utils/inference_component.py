from typing import Optional

import torch.nn as nn

from modalities.inference.config import DeviceMode


class InferenceComponent:
    def __init__(self, model: nn.Module, device_mode: DeviceMode, gpu_id: Optional[int] = None) -> None:
        if device_mode == DeviceMode.CPU:
            self.model = self.model.cpu()
        elif device_mode == DeviceMode.SINGLE_GPU:
            self.model = self.model.cuda(gpu_id)
        else:
            self.model = model
        model.eval()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
