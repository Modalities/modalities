import os
from contextlib import contextmanager
from functools import wraps
from typing import Any


@contextmanager
def temporary_env_var(key, value):
    """
    Temporarily set an environment variable.

    Args:
        key (str): The environment variable name.
        value (str): The temporary value to set.
    """
    original_value = os.environ.get(key)  # Store the original value (if any)
    os.environ[key] = value  # Set the temporary value
    try:
        yield  # Allow code execution within the context
    finally:
        # Restore the original value or delete the key if it wasn't set originally
        if original_value is None:
            del os.environ[key]
        else:
            os.environ[key] = original_value


def temporary_env_vars_decorator(env_vars: dict[str, Any]):
    """
    Decorator to temporarily set multiple environment variables for the duration of a function call.

    Args:
        env_vars (dict): A dictionary of environment variable names and their temporary values.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            original_values = {}  # Store original values of environment variables
            try:
                # Set the temporary environment variables
                for key, value in env_vars.items():
                    original_values[key] = os.environ.get(key)  # Save original value
                    os.environ[key] = value  # Set temporary value
                return func(*args, **kwargs)  # Execute the decorated function
            finally:
                # Restore original values or delete keys if not originally set
                for key, original_value in original_values.items():
                    if original_value is None:
                        del os.environ[key]
                    else:
                        os.environ[key] = original_value

        return wrapper

    return decorator
