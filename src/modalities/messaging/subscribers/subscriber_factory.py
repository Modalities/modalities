from pathlib import Path
from typing import List

from modalities.config.config import WandbMode
from modalities.dataloader.dataloader import LLMDataLoader
from modalities.messaging.subscribers.batch_progress_subscriber import DummyProgressSubscriber, RichProgressSubscriber
from modalities.messaging.subscribers.results_subscriber import (
    DummyResultSubscriber,
    RichResultSubscriber,
    WandBEvaluationResultSubscriber,
)


class ProgressSubscriberFactory:
    @staticmethod
    def get_rich_progress_subscriber(
        train_dataloader: LLMDataLoader,
        eval_dataloaders: List[LLMDataLoader],
        global_num_seen_steps: int,
        local_rank: int,
    ) -> RichProgressSubscriber:
        if local_rank == 0:
            train_split_num_steps = {
                train_dataloader.dataloader_tag: (len(train_dataloader) + global_num_seen_steps, global_num_seen_steps)
            }

            eval_splits_num_steps = {dataloader.dataloader_tag: len(dataloader) for dataloader in eval_dataloaders}

            subscriber = RichProgressSubscriber(train_split_num_steps, eval_splits_num_steps)
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
        config_file_path: Path,
        directory: Path = None,
    ) -> WandBEvaluationResultSubscriber:
        if local_rank == 0 and (mode != WandbMode.DISABLED):
            result_subscriber = WandBEvaluationResultSubscriber(
                project, experiment_id, mode, directory, config_file_path
            )
        else:
            result_subscriber = ResultsSubscriberFactory.get_dummy_result_subscriber()
        return result_subscriber