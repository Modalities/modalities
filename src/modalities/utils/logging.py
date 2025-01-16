import logging


def get_logger(name: str = "main") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    return logger
