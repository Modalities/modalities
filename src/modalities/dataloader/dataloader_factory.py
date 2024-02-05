from torch.utils.data.dataset import Dataset

from modalities.config.config import DataLoaderConfig, DatasetConfig
from modalities.config.resolver_register import ResolverRegister
from modalities.dataloader.dataloader import LLMDataLoader
from modalities.dataloader.open_gptx_dataset.open_gptx_dataset import OpenGPTXMMapDataset
from modalities.dataloader.samplers import ResumableBatchSampler


class OpenGPTXDatasetWrapper(Dataset):
    def __init__(self, open_gptx_dataset: OpenGPTXMMapDataset, num_samples: int) -> None:
        super().__init__()
        self.open_gptx_dataset = open_gptx_dataset
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        if self.num_samples > idx:
            return self.open_gptx_dataset.__getitem__(idx)
        else:
            raise ValueError("num_samples <= idx")


class DataloaderFactory:
    @staticmethod
    def get_dataloader(
        resolvers: ResolverRegister, config: DataLoaderConfig, skip_num_batches: int = 0
    ) -> LLMDataLoader:
        # TODO: replace this with dynamic nested object instantiation. (More details: Different Dataloaders require
        #  different objects in their constructors. the resolvers should be able to provide the necessary complex
        #  objects automatically, without us manually creating this complex factory.)
        additional_init_payload = {}
        if hasattr(config.config.dataset.config, "tokenizer"):
            tokenizer = resolvers.build_component_by_config(config=config.config.dataset.config.tokenizer)
            tokenizer.pad_token = tokenizer.eos_token
            additional_init_payload.update(tokenizer=tokenizer)

        dataset = resolvers.build_component_by_config(
            config=config.config.dataset, extra_kwargs=additional_init_payload
        )

        # BUG: Sometimes the dataset genereated by the OpenGPTXMMap implementation has too many samples.
        # This is a workaround to fix the dataset to the size, as specified in the config!
        # TODO: Fix the OpenGPTX implementation and get rid of this hack.
        if isinstance(config.config.dataset.config, DatasetConfig.OpenGPTXMMapDatasetConfig):
            dataset = OpenGPTXDatasetWrapper(
                open_gptx_dataset=dataset, num_samples=config.config.dataset.config.num_samples
            )

        collator = resolvers.build_component_by_config(config=config.config.collate_fn)
        sampler = resolvers.build_component_by_config(
            config=config.config.batch_sampler.config.sampler, extra_kwargs=dict(dataset=dataset)
        )

        batch_sampler = resolvers.build_component_by_config(
            config=config.config.batch_sampler,
            extra_kwargs=dict(
                sampler=sampler,
            ),
        )

        resumable_batch_sampler = ResumableBatchSampler(
            start_index=skip_num_batches, underlying_batch_sampler=batch_sampler
        )

        dataloader = resolvers.build_component_by_config(
            config=config,
            extra_kwargs=dict(
                dataset=dataset,
                batch_sampler=resumable_batch_sampler,
                collate_fn=collator,
            ),
        )

        # TODO we should have this check rather in the gym. Here, it is clear that
        # we are using the LLMDataLoader
        assert isinstance(
            dataloader, LLMDataLoader
        ), f"Dataloader Class must use the {LLMDataLoader.__name__}-Interface"
        return dataloader
