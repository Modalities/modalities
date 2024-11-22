from enum import Enum


class Component_V_W_X_IF:
    def print(self) -> None:
        print("ComponentIF")


# Dependencies


class ComponentV(Component_V_W_X_IF):
    def __init__(self, val_v: str) -> None:
        self.val_v = val_v


class ComponentW(Component_V_W_X_IF):
    def __init__(self, val_w: str) -> None:
        self.val_w = val_w


# Components


class ComponentX(Component_V_W_X_IF):
    def __init__(self, val_x: str, single_dependency: Component_V_W_X_IF) -> None:
        self.val_x = val_x
        self.single_dependency = single_dependency


class ComponentY:
    def __init__(self, val_y: str, multi_dependency: list[Component_V_W_X_IF]) -> None:
        self.val_y = val_y
        self.multi_dependency = multi_dependency


class ComponentZ:
    def __init__(self, val_z: str) -> None:
        self.val_z = val_z


class ComponentTypes(Enum):
    COMP_V = ComponentV
    COMP_W = ComponentW
    COMP_X = ComponentX
    COMP_Y = ComponentY
    COMP_Z = ComponentZ
