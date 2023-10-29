from llm_gym.forward_pass import ModelInferenceComponent
import torch.cuda.nccl as nccl
from pkg_resources import packaging
import os
from llm_gym.gpt2.collator import GPT2LLMCollator, LMWikiBookCorpusDatasetFactory
from llm_gym.gpt2.gpt2_model import GPT2LLM, NNModel
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
import time
from llm_gym.loss_functions import CLMCrossEntropyLoss


# global flag that confirms ampere architecture, cuda version and
# nccl version to verify bfloat16 native support is ready


def has_bfloat_support():
    return (
        torch.version.cuda
        and torch.cuda.is_bf16_supported()
        and packaging.version.parse(torch.version.cuda).release >= (11, 0)
        and dist.is_nccl_available()
        and nccl.version() >= (2, 10)
    )


# requires grad scaler in main loop
fpSixteen = MixedPrecision(
    param_dtype=torch.float16,
    # Gradient communication precision.
    reduce_dtype=torch.float16,
    # Buffer precision.
    buffer_dtype=torch.float16,
)

bfSixteen = MixedPrecision(
    param_dtype=torch.bfloat16,
    # Gradient communication precision.
    reduce_dtype=torch.bfloat16,
    # Buffer precision.
    buffer_dtype=torch.bfloat16,
)

bfSixteen_working = MixedPrecision(
    param_dtype=torch.float32,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

fp32_policy = MixedPrecision(
    param_dtype=torch.float32,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float32,
)


def setup():
    # initialize the process group
    dist.init_process_group("nccl")


def cleanup():
    dist.destroy_process_group()


def setup_model() -> NNModel:
    model = GPT2LLM(prediction_publication_key="logits")
    return model



def fsdp_main(train_batch_size: int, test_batch_size, lr: int, gamma: int, dataset_path: str,
              track_memory: bool, epochs: int, run_validation: bool, save_model: bool):

    model = setup_model()

    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    # create dataset
    dataset_dict = LMWikiBookCorpusDatasetFactory.construct(dataset_path)
    train_dataset = dataset_dict["train"]
    val_dataset = dataset_dict["validation"]

    sampler_train = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True)
    sampler_val = DistributedSampler(val_dataset, rank=rank, num_replicas=world_size)

    setup()

    cuda_kwargs = {'num_workers': 2,
                   'pin_memory': True,
                   'shuffle': False}
    pad_to_multiple_of = 8
    tokenizer_file_path = "/raid/s3/opengptx/max_lue/LLMgym/src/llm_gym/gpt2/tokenizer.json"
    collate_fn = GPT2LLMCollator(target_publication_key="target_key", tokenizer_file_path=tokenizer_file_path,
                                 pad_to_multiple_of=pad_to_multiple_of)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, sampler=sampler_train, **cuda_kwargs,
                                               collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=test_batch_size,
                                             sampler=sampler_val, **cuda_kwargs, collate_fn=collate_fn)

    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    torch.cuda.set_device(local_rank)

    if has_bfloat_support():
        mp_policy = bfSixteen
    else:
        mp_policy = None  # defaults to fp32

    # model is on CPU before input to FSDP
    model = FSDP(model,
                 auto_wrap_policy=None,
                 mixed_precision=mp_policy,
                 sharding_strategy=sharding_strategy,
                 device_id=torch.cuda.current_device())

    model_inference_component = ModelInferenceComponent(model=model, post_processors=[])

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    loss_fun = CLMCrossEntropyLoss(target_subscription_key="target_key", prediction_subscription_key="logits")

    best_val_loss = float("inf")
    curr_val_loss = float("inf")
    file_save_name = "gpt2-model-"

    if rank == 0:
        time_of_run = get_date_of_run()
        dur = []
        train_acc_tracking = []
        val_acc_tracking = []

    if rank == 0 and track_memory:
        mem_alloc_tracker = []
        mem_reserved_tracker = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_accuracy = train(model_inference_component, rank, world_size, train_loader, optimizer, epoch, loss_fun)
        if run_validation:
            curr_val_loss = validation(model_inference_component, rank, val_loader, loss_fun)
        scheduler.step()

        if rank == 0:

            print(f"--> epoch {epoch} completed...entering save and stats zone")

            dur.append(time.time() - t0)
            train_acc_tracking.append(train_accuracy.item())

            if run_validation:
                val_acc_tracking.append(curr_val_loss.item())

            if track_memory:
                mem_alloc_tracker.append(
                    format_metrics_to_gb(torch.cuda.memory_allocated())
                )
                mem_reserved_tracker.append(
                    format_metrics_to_gb(torch.cuda.memory_reserved())
                )
            print("completed save and stats zone...")

        if save_model and curr_val_loss < best_val_loss:

            # save
            if rank == 0:
                print("--> entering save model state")

            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
                cpu_state = model.state_dict()
            # print(f"saving process: rank {rank}  done w state_dict")

            if rank == 0:
                print("--> saving model ...")
                currEpoch = ("-" + str(epoch) + "-" + str(round(curr_val_loss.item(), 4)) + ".pt")
                print(f"--> attempting to save model prefix {currEpoch}")
                save_name = file_save_name + "-" + time_of_run + "-" + currEpoch
                print(f"--> saving as model name {save_name}")

                torch.save(cpu_state, save_name)

        if curr_val_loss < best_val_loss:

            best_val_loss = curr_val_loss
            if rank == 0:
                print(f"-->>>> New Val Loss Record: {best_val_loss}")

    dist.barrier()
    cleanup()


if __name__ == '__main__':

    fsdp_main(train_batch_size=32, test_batch_size=32, lr=1.e-4, gamma=0.1, dataset_path="/raid/s3/opengptx/max_lue/LLMgym/src/llm_gym/gpt2/data/wikitext-103-raw-v1-tokenized",
              track_memory=False, epochs=2, run_validation=True, save_model=True)
