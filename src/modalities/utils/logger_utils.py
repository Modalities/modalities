import logging

import torch


def get_logger(name: str = "main") -> logging.Logger:
    rank_info = ""

    if torch.distributed.is_initialized():
        rank_info = f"[RANK {torch.distributed.get_rank()}] "

    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(f"{rank_info}%(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    return logger
