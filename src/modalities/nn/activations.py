from pydantic import BaseModel

class ActivationConfig(BaseModel):
    pass

class LeakyReLUConfig(ActivationConfig):
    scale: float
    negative_slope: float = 0.01
    zero_point: int