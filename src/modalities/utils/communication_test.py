import os
import sys

import torch
import torch.distributed as dist


def run_communication_test():
    """Test the all_gather communication operation in a distributed setting."""
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    # Each rank creates a tensor of shape [4], filled with its rank ID
    local_tensor = torch.full((4,), fill_value=rank, dtype=torch.int32, device=device)

    # Prepare gather list
    gathered = [torch.empty_like(local_tensor) for _ in range(world_size)]

    # Run all_gather
    dist.all_gather(gathered, local_tensor)

    # Verify contents
    expected = [torch.full((4,), fill_value=i, dtype=torch.int32, device=device) for i in range(world_size)]
    errors = [not torch.equal(g, e) for g, e in zip(gathered, expected)]

    if any(errors):
        print(f"[Rank {rank}] ❌ Allgather verification FAILED")
        for i, (g, e) in enumerate(zip(gathered, expected)):
            if not torch.equal(g, e):
                print(f"[Rank {rank}] Mismatch at slot {i}: got {g.cpu().tolist()}, expected {e.cpu().tolist()}")
        dist.destroy_process_group()
        sys.exit(1)  # Exit with error
    else:
        print(f"[Rank {rank}] ✅ Allgather verification PASSED")
