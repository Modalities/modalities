import os
from pathlib import Path
from typing import Optional

from modalities.config.config import WandbMode
from modalities.dataloader.dataloader import LLMDataLoader
from modalities.logging_broker.subscriber_impl.progress_subscriber import (
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
        eval_dataloaders: list[LLMDataLoader],
        train_dataloader_tag: str,
        num_seen_steps: int,
        num_target_steps: int,
        global_rank: int,
    ) -> RichProgressSubscriber:
        if global_rank == 0:
            train_split_num_steps = {
                # first element describes the total number of steps
                # and the second element describes the number of steps already completed
                train_dataloader_tag: (num_target_steps, num_seen_steps)
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
    def get_rich_result_subscriber(num_ranks: int, global_rank: int) -> RichResultSubscriber:
        if global_rank == 0:
            return RichResultSubscriber(num_ranks)
        else:
            return ResultsSubscriberFactory.get_dummy_result_subscriber()

    @staticmethod
    def get_dummy_result_subscriber() -> DummyResultSubscriber:
        return DummyResultSubscriber()

    @staticmethod
    def get_wandb_result_subscriber(
        global_rank: int,
        project: str,
        experiment_id: str,
        mode: WandbMode,
        config_file_path: Path,
        directory: Optional[Path] = None,
    ) -> WandBEvaluationResultSubscriber:
        if global_rank == 0 and (mode != WandbMode.DISABLED):
            if directory is not None:
                # we store cache, data and offline runs under directory
                absolute_dir = directory.absolute()
                (absolute_dir / "wandb").mkdir(parents=True, exist_ok=True)

                os.environ["WANDB_CACHE_DIR"] = str(absolute_dir)
                os.environ["WANDB_DIR"] = str(absolute_dir)

                # see https://community.wandb.ai/t/wandb-artifact-cache-directory-fills-up-the-home-directory/5224/5
                # and https://github.com/wandb/wandb/issues/6792
                os.environ["WANDB_DATA_DIR"] = str(absolute_dir)
                os.environ["WANDB_ARTIFACT_LOCATION"] = str(absolute_dir)
                os.environ["WANDB_ARTIFACT_DIR"] = str(absolute_dir)
                os.environ["WANDB_CONFIG_DIR"] = str(absolute_dir)
            else:
                absolute_dir = None

            result_subscriber = WandBEvaluationResultSubscriber(
                project, experiment_id, mode, absolute_dir, config_file_path
            )
        else:
            result_subscriber = ResultsSubscriberFactory.get_dummy_result_subscriber()
        return result_subscriber
