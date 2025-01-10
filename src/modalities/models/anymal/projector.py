import torch
from pydantic import BaseModel

from modalities.models.model import NNModel


class MLPProjectorConfig(BaseModel):
    """
    Args:
        input_dim (`int`):
            input dimension
        output_dim (`int`):
            output dimension
    """

    input_dim: int
    output_dim: int


class MLPProjector(NNModel):
    """
    Two-layer MLP
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
    ) -> None:
        """
        Args:
            input_dim (`int`):
                input dimension
            output_dim (`int`):
                output dimension
        """
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, output_dim, bias=True)
        self.act = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(output_dim, output_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
