import os
from typing import Dict
from llm_gym.env_utils import has_bfloat_support, bfSixteen
from llm_gym.forward_pass import ModelForwardPass
from llm_gym.gpt2.gpt2_model import NNModel, GPT2LLM
from llm_gym.gpt2.collator import GPT2LLMCollator, LMWikiBookCorpusDatasetFactory
from llm_gym.gym import ResultsCallback, RichProgressCallback, Trainer
from llm_gym.loss_functions import CLMCrossEntropyLoss
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


class Main:

    def __init__(self, dataset_path: str, num_epochs: int) -> None:
        dist.init_process_group("nccl")

        self.local_rank = int(os.environ['LOCAL_RANK'])
        self.global_rank = int(os.environ['RANK'])
        self.world_size = int(os.environ['WORLD_SIZE'])
        self.dataset_path = dataset_path
        self.num_epochs = num_epochs

        self.model = GPT2LLM(prediction_publication_key="logits")
        self.wrapped_model = self.wrap_fsdp_model(self.model)
        self.model_forward_pass = ModelForwardPass(model=self.wrapped_model, post_processors=[])

        self.optimizer = optim.AdamW(self.wrapped_model.parameters(), lr=0.0001)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.1)

        self.loss_fun = CLMCrossEntropyLoss(target_subscription_key="target_key", prediction_subscription_key="logits")

        # data loaders
        self.data_loaders = self.create_dataloaders(train_batch_size=8, test_batch_size=8)  # TODO make dynamic
        train_split_length = {split_key: len(split) for split_key, split in self.data_loaders.items() if split_key == "train"}

        batch_processed_callback = RichProgressCallback(subscribing_global_rank=self.global_rank,
                                                        num_epochs=self.num_epochs, split_lengths=train_split_length)
        results_callback = ResultsCallback(subscribing_global_rank=self.global_rank)
        self.trainer = Trainer(local_rank=self.local_rank, global_rank=self.global_rank, batch_processed_callback=batch_processed_callback,
                               results_callback=results_callback)


    def run(self):
        for current_epoch in range(self.num_epochs):
            self.trainer.train_epoch(model_forward_pass=self.model_forward_pass, train_loader=self.data_loaders["train"], loss_fun=self.loss_fun,
                                     optimizer=self.optimizer)

    def wrap_fsdp_model(self, model: NNModel) -> FSDP:
        sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
        torch.cuda.set_device(self.local_rank)

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
        return model

    def create_dataloaders(self, train_batch_size: int, test_batch_size: int) -> Dict[str, DataLoader]:
        # create dataset splits
        dataset_dict = LMWikiBookCorpusDatasetFactory.construct(self.dataset_path)
        train_dataset = dataset_dict["train"]
        val_dataset = dataset_dict["validation"]

        # create samplers
        sampler_train = DistributedSampler(train_dataset, rank=self.global_rank, num_replicas=self.world_size, shuffle=True)
        sampler_val = DistributedSampler(val_dataset, rank=self.global_rank, num_replicas=self.world_size)

        # create dataloaders
        cuda_kwargs = {'num_workers': 2,
                       'pin_memory': True,
                       'shuffle': False}
        pad_to_multiple_of = 8
        tokenizer_file_path = "/raid/s3/opengptx/max_lue/LLMgym/src/llm_gym/gpt2/tokenizer.json"
        collate_fn = GPT2LLMCollator(target_publication_key="target_key", tokenizer_file_path=tokenizer_file_path,
                                     pad_to_multiple_of=pad_to_multiple_of)
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, sampler=sampler_train, **cuda_kwargs,
                                  collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=test_batch_size,
                                sampler=sampler_val, **cuda_kwargs, collate_fn=collate_fn)

        return {"train": train_loader, "val": val_loader}


if __name__ == '__main__':
    dataset_path = "/raid/s3/opengptx/max_lue/LLMgym/src/llm_gym/gpt2/data/wikitext-103-raw-v1-tokenized"

    main = Main(dataset_path=dataset_path, num_epochs=3)
    main.run()
