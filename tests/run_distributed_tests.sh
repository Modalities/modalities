#!/bin/sh

# Run pytest with torchrun

# test_fsdp_to_disc_checkpointing
CUDA_VISIBLE_DEVICES=0,1 torchrun --rdzv-endpoint localhost:29502 --nnodes 1 --nproc_per_node 2 $(which pytest) checkpointing/test_fsdp_to_disc_checkpointing.py

# test_fsdp_warmstart
CUDA_VISIBLE_DEVICES=0,1 torchrun --rdzv-endpoint localhost:29502 --nnodes 1 --nproc_per_node 2 $(which pytest) end2end_tests/test_fsdp_warmstart.py -k "test_warm_start"
CUDA_VISIBLE_DEVICES=0,1 torchrun --rdzv-endpoint localhost:29502 --nnodes 1 --nproc_per_node 2 $(which pytest) end2end_tests/test_fsdp_warmstart.py -k "test_warmstart_dataloader"


# test_distributed_repeating_dataloader
CUDA_VISIBLE_DEVICES=0,1 torchrun --rdzv-endpoint localhost:29502 --nnodes 1 --nproc_per_node 2 $(which pytest) dataloader/distributed/test_distributed_repeating_dataloader.py -k "test_resumable_dataloader_without_shuffling"

# test_distributed_dataloader
CUDA_VISIBLE_DEVICES=0,1 torchrun --rdzv-endpoint localhost:29502 --nnodes 1 --nproc_per_node 2 $(which pytest) dataloader/distributed/test_distributed_dataloader.py -k "test_resumable_dataloader_without_shuffling"
CUDA_VISIBLE_DEVICES=0,1 torchrun --rdzv-endpoint localhost:29502 --nnodes 1 --nproc_per_node 2 $(which pytest) dataloader/distributed/test_distributed_dataloader.py -k "test_resumable_dataloader_with_shuffling_without_skipping"
CUDA_VISIBLE_DEVICES=0,1 torchrun --rdzv-endpoint localhost:29502 --nnodes 1 --nproc_per_node 2 $(which pytest) dataloader/distributed/test_distributed_dataloader.py -k "test_resumable_dataloader_with_shuffling_and_skipped_batches"

