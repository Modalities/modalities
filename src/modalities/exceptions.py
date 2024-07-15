class DatasetNotFoundError(Exception):
    pass


class BatchStateError(Exception):
    pass


class CheckpointingError(Exception):
    pass


class RunningEnvError(Exception):
    pass


class TimeRecorderStateError(Exception):
    pass


class OptimizerError(Exception):
    pass


class ConfigError(Exception):
    pass
