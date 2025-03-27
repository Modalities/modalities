from torch.distributed.fsdp import FSDPModule as FSDP2
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP1

FSDPX = FSDP1 | FSDP2
