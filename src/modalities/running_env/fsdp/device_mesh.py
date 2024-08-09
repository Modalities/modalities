from pydantic import BaseModel
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh


class DeviceMeshConfig(BaseModel):
    device_type: str


def get_device_mesh(device_type: str) -> DeviceMesh:
    dims = []
    names = []
    for d, name in zip([1, 2, 1], ["pp", "dp", "tp"], strict=True):
        if d > 1:
            dims.append(d)
            names.append(name)
    names = tuple(names)
    return init_device_mesh(device_type, dims, mesh_dim_names=names)
