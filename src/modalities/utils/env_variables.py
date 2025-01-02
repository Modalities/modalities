import os
from contextlib import contextmanager

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