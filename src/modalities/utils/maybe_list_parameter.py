import inspect
from functools import wraps
from typing import Callable, Concatenate, ParamSpec, TypeAlias, TypeVar, cast

T = TypeVar("T")  # Represents the parameter we want to wrap into a list.
P = ParamSpec("P")  # Represents the other parameters of the function.
R1 = TypeVar("R1")  # Represents the return type of the base function.
R2 = TypeVar("R2")  # Represents the return type of the additional reducers.

ListResultsReducer: TypeAlias = Callable[[list[R1]], R2]
ListInputAndResultsReducer: TypeAlias = Callable[[list[T], list[R1]], R2]

BaseFunc: TypeAlias = Callable[Concatenate[T, P], R1]
MaybeListWrappedFunc: TypeAlias = Callable[Concatenate[T | list[T], P], R1 | list[R1] | R2]

MaybeListDecorator: TypeAlias = Callable[[BaseFunc[T, P, R1]], MaybeListWrappedFunc[T, P, R1, R2]]


def maybe_list_parameter(
    parameter_name: str,
    apply_to_list_result: ListResultsReducer[R1, R2] | None = None,
    apply_to_list_input_and_result: ListInputAndResultsReducer[T, R1, R2] | None = None,
) -> MaybeListDecorator[T, P, R1, R2]:
    """Decorator factory allowing a specific parameter to be a single item or a list.
    If a list is provided, the wrapped function is called once per element and a list
    of results is returned; otherwise the single result is returned.

    Args:
        parameter_name (str): The name of the parameter to treat as a list or single item.
        apply_to_list_result (ListResultsReducer | None): Reduces list of results -> single R
        apply_to_list_input_and_result (ListInputAndResultsReducer | None):
            Takes (original list input, list of results) -> single R
    (mutually exclusive)
    """

    def decorator(func: BaseFunc[T, P, R1]) -> MaybeListWrappedFunc[T, P, R1, R2]:
        sig = inspect.signature(func)

        # Find positional index of the target parameter (if present)
        param_pos_index: int | None = None
        for idx, (name, _) in enumerate(sig.parameters.items()):
            if name == parameter_name:
                param_pos_index = idx
                break
        if param_pos_index is None:
            raise ValueError(f"Parameter '{parameter_name}' not found in function '{func.__name__}' signature.")

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R1 | list[R1] | R2:
            # Obtain value (positional or kw)
            if parameter_name in kwargs:
                param_value: T | list[T] = kwargs[parameter_name]
            elif param_pos_index < len(args):
                param_value = args[param_pos_index]
            else:
                # Parameter not supplied; just call through
                return func(*args, **kwargs)

            # If not a list, call directly
            if not isinstance(param_value, list):
                return func(*args, **kwargs)

            # Process each element
            results: list[R1] = []
            for item in param_value:
                if parameter_name in kwargs:
                    new_kwargs = dict(kwargs)
                    new_kwargs[parameter_name] = item
                    results.append(func(*args, **new_kwargs))
                else:
                    new_args = list(args)
                    new_args[param_pos_index] = item
                    results.append(func(*new_args, **kwargs))

            if apply_to_list_result is not None:
                if apply_to_list_input_and_result is not None:
                    raise ValueError("Cannot provide both apply_to_list_result and apply_to_list_input_and_result.")
                return apply_to_list_result(results)
            if apply_to_list_input_and_result is not None:
                return apply_to_list_input_and_result(param_value, results)
            return results

        return cast(MaybeListWrappedFunc, wrapper)

    return decorator
