from llm_gym.config.config import DataLoaderConfig
from llm_gym.dataloader.dataloader import LLMDataLoader
from llm_gym.dataloader.samplers import ResumableBatchSampler
from llm_gym.resolver_register import ResolverRegister


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
