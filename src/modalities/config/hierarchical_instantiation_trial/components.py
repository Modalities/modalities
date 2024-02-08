from enum import Enum


class CompV:
    def __init__(self, val_0: str) -> None:
        self.val_0 = val_0


class CompW:
    def __init__(self, val_0: str) -> None:
        self.val_0 = val_0


class CompX:
    def __init__(self, val_1: str) -> None:
        self.val_1 = val_1


class CompY:
    def __init__(self, val_2: str, comp_x: CompX) -> None:
        self.val_2 = val_2
        self.comp_x = comp_x


class CompZ:
    def __init__(self, val_3: str, comp_y: CompY) -> None:
        self.val_3 = val_3
        self.comp_y = comp_y


class ComponentTypes(Enum):
    COMP_V = CompV
    COMP_W = CompW
    COMP_X = CompX
    COMP_Y = CompY
    COMP_Z = CompZ
