import os
import torch
import torch.distributed as dist

def main():
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    rank = dist.get_rank()
    print(f"{rank=}")
    world = dist.get_world_size()
    assert world == 2, f"Run with exactly 2 ranks, got {world}"

    if torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())

    if rank == 0:
        tensor = torch.arange(5, dtype=torch.int64).to("cuda")
        dist.send(tensor, dst=1)
        print(f"[rank0] sent: {tensor}")
    else:
        recv = torch.empty(5, dtype=torch.int64).to("cuda")
        dist.recv(recv, src=0)
        print(f"[rank1] received: {recv}")

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()