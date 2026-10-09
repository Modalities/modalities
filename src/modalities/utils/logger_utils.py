import logging

try:
    # Only needed for the rank prefix. Guarded so that CPU-only tooling -- data
    # preprocessing in particular -- can import the dataloader utilities without
    # pulling in torch, which modalities declares in its cpu/cu12x extras rather
    # than as a base dependency.
    import torch
except ImportError:  # pragma: no cover - exercised only in torch-free installs
    torch = None


def get_logger(name: str = "main") -> logging.Logger:
    rank_info = ""

    if torch is not None and torch.distributed.is_initialized():
        rank_info = f"[RANK {torch.distributed.get_rank()}] "

    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(f"{rank_info}%(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    return logger
