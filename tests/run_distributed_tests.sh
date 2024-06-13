#!/bin/sh

#################
### VARIABLES ###
#################
DEV0=3
DEV1=4
COVERAGE=--no-cov


#############
### TESTS ###
#################
# test_fsdp_to_disc_checkpointing
CUDA_VISIBLE_DEVICES=$DEV0,$DEV1 torchrun --rdzv-endpoint localhost:29502 --nnodes 1 --nproc_per_node 2 $(which pytest) checkpointing/test_fsdp_to_disc_checkpointing.py $COVERAGE

# test_fsdp_warmstart
CUDA_VISIBLE_DEVICES=$DEV0,$DEV1 torchrun --rdzv-endpoint localhost:29502 --nnodes 1 --nproc_per_node 2 $(which pytest) end2end_tests/test_fsdp_warmstart.py -k "test_warm_start" $COVERAGE
CUDA_VISIBLE_DEVICES=$DEV0,$DEV1 torchrun --rdzv-endpoint localhost:29502 --nnodes 1 --nproc_per_node 2 $(which pytest) end2end_tests/test_fsdp_warmstart.py -k "test_warmstart_dataloader" $COVERAGE


# test_distributed_repeating_dataloader
CUDA_VISIBLE_DEVICES=$DEV0,$DEV1 torchrun --rdzv-endpoint localhost:29502 --nnodes 1 --nproc_per_node 2 $(which pytest) dataloader/distributed/test_distributed_repeating_dataloader.py -k "test_resumable_dataloader_without_shuffling" $COVERAGE

# test_distributed_dataloader
CUDA_VISIBLE_DEVICES=$DEV0,$DEV1 torchrun --rdzv-endpoint localhost:29502 --nnodes 1 --nproc_per_node 2 $(which pytest) dataloader/distributed/test_distributed_dataloader.py -k "test_resumable_dataloader_without_shuffling" $COVERAGE
CUDA_VISIBLE_DEVICES=$DEV0,$DEV1 torchrun --rdzv-endpoint localhost:29502 --nnodes 1 --nproc_per_node 2 $(which pytest) dataloader/distributed/test_distributed_dataloader.py -k "test_resumable_dataloader_with_shuffling_without_skipping" $COVERAGE
CUDA_VISIBLE_DEVICES=$DEV0,$DEV1 torchrun --rdzv-endpoint localhost:29502 --nnodes 1 --nproc_per_node 2 $(which pytest) dataloader/distributed/test_distributed_dataloader.py -k "test_resumable_dataloader_with_shuffling_and_skipped_batches" $COVERAGE

# test optimizer
CUDA_VISIBLE_DEVICES=$DEV0 torchrun --rdzv-endpoint localhost:29502 --nnodes 1 --nproc_per_node 1 $(which pytest) test_optimizer_factory.py $COVERAGE

# test initialization
CUDA_VISIBLE_DEVICES=$DEV0 torchrun --rdzv-endpoint localhost:29502 --nnodes 1 --nproc_per_node 1 $(which pytest) test_initialization.py $COVERAGE

