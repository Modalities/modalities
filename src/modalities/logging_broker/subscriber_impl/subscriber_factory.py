from pathlib import Path
from typing import Dict, List

from modalities.config.config import WandbMode
from modalities.dataloader.dataloader import LLMDataLoader
from modalities.logging_broker.subscriber_impl.batch_progress_subscriber import (
    DummyProgressSubscriber,
    RichProgressSubscriber,
)
from modalities.logging_broker.subscriber_impl.results_subscriber import (
    DummyResultSubscriber,
    RichResultSubscriber,
    WandBEvaluationResultSubscriber,
)


class ProgressSubscriberFactory:
    @staticmethod
    def get_rich_progress_subscriber(
        train_dataloader: LLMDataLoader,
        eval_dataloaders: List[LLMDataLoader],
        world_size: int,
        global_num_seen_samples: int,
        local_rank: int,
    ) -> RichProgressSubscriber:
        if local_rank == 0:
            skip_num_local_train_batches = global_num_seen_samples // world_size // train_dataloader.batch_size
            train_split_num_samples = {
                train_dataloader.dataloader_tag: (len(train_dataloader) + skip_num_local_train_batches)
                * world_size
                * train_dataloader.batch_size
            }

            eval_splits_num_samples = {
                dataloader.dataloader_tag: len(dataloader) * world_size * dataloader.batch_size
                for dataloader in eval_dataloaders
            }

            subscriber = RichProgressSubscriber(world_size, train_split_num_samples, eval_splits_num_samples)
        else:
            subscriber = ProgressSubscriberFactory.get_dummy_progress_subscriber()
        return subscriber

    @staticmethod
    def get_dummy_progress_subscriber() -> DummyProgressSubscriber:
        return DummyProgressSubscriber()


class ResultsSubscriberFactory:
    @staticmethod
    def get_rich_result_subscriber(num_ranks: int, local_rank: int) -> RichResultSubscriber:
        if local_rank == 0:
            return RichResultSubscriber(num_ranks)
        else:
            return ResultsSubscriberFactory.get_dummy_result_subscriber()

    @staticmethod
    def get_dummy_result_subscriber() -> DummyResultSubscriber:
        return DummyResultSubscriber()

    @staticmethod
    def get_wandb_result_subscriber(
        local_rank: int,
        project: str,
        experiment_id: str,
        mode: WandbMode,
        directory: Path = None,
        experiment_config: Dict = None,
    ) -> WandBEvaluationResultSubscriber:
        if local_rank == 0 and (mode == WandbMode.ONLINE or mode == WandbMode.OFFLINE):
            result_subscriber = WandBEvaluationResultSubscriber(
                project, experiment_id, mode, directory, experiment_config
            )
        else:
            result_subscriber = ResultsSubscriberFactory.get_dummy_result_subscriber()
        return result_subscriber
