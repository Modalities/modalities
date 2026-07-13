import torch.nn as nn


def has_tied_word_embeddings(model: nn.Module) -> bool:
    model_has_tied_word_embeddings = getattr(model, "has_tied_word_embeddings", None)
    if model_has_tied_word_embeddings is None:
        raise TypeError(
            f"{type(model).__name__} must define 'has_tied_word_embeddings' to be used with tied-embedding validation."
        )

    return bool(model_has_tied_word_embeddings)
