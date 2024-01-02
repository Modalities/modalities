from datetime import datetime

from torch import distributed as dist


def get_date_of_run():
    """create date and time for file save uniqueness
    example: 2022-05-07__14-31-22'
    """
    date_of_run = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    return date_of_run


def format_metrics_to_gb(item):
    """quick function to format numbers to gigabyte and round to 4 digit precision"""
    g_gigabyte = 1024**3
    metric_num = item / g_gigabyte
    metric_num = round(metric_num, ndigits=4)
    return metric_num


class DistributedSetupInfo:
    @property
    def dist_launched(self) -> bool:
        return dist.is_initialized() and dist.is_torchelastic_launched()

    @property
    def rank(self) -> int:
        if not dist.is_initialized():
            return 0
        return dist.get_rank()


dist_setup_info = DistributedSetupInfo()
