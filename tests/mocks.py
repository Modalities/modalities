class MockMeshEntry:
    def __init__(self, coordinate: int, size: int):
        self._coordinate = coordinate
        self._size = size

    def get_coordinate(self) -> list[int]:
        return [self._coordinate]

    def size(self):
        return self._size


class MockDeviceMesh(dict[str, MockMeshEntry]):
    def __init__(self, mesh_dict: dict[str, tuple[int, int]]):
        super().__init__()
        for key, (coord, size) in mesh_dict.items():
            self[key] = MockMeshEntry(coord, size)
        self.mesh_dim_names = list(mesh_dict.keys())
