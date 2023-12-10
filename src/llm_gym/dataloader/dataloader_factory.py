from llm_gym.config.config import DataLoaderConfig
from llm_gym.resolver_register import ResolverRegister
from llm_gym.dataloader.dataloader import LLMDataLoader

class DataloaderFactory:
    @staticmethod
    def get_dataloader(resolvers: ResolverRegister, config: DataLoaderConfig) -> LLMDataLoader:
        dataset = resolvers.build_component_by_config(config=config.config.dataset)
        collator = resolvers.build_component_by_config(config=config.config.collate_fn)
        sampler = resolvers.build_component_by_config(config=config.config.sampler, extra_kwargs=dict(dataset=dataset))
        dataloader = resolvers.build_component_by_config(
            config=config,
            extra_kwargs=dict(
                dataset=dataset,
                sampler=sampler,
                collate_fn=collator,
            ),
        )
        return dataloader
