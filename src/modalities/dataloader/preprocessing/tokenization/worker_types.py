from enum import Enum


class WorkerTypes(Enum):
    POPULATOR = "POPULATOR"
    READER = "READER"
    TOKENIZER = "TOKENIZER"
    WRITER = "WRITER"
    LOGGING = "LOGGING"
