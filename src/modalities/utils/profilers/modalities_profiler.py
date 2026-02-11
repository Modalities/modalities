import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
from pydantic import BaseModel
from tqdm import trange

from modalities.config.config import ProcessGroupBackendType
from modalities.config.pydantic_if_types import PydanticSteppableComponentIFType, PydanticSteppableProfilerIFType
from modalities.main import Main
from modalities.running_env.cuda_env import CudaEnv
from modalities.util import get_experiment_id_from_config, get_synced_experiment_id_of_run
from modalities.utils.logger_utils import get_logger

logger = get_logger("modalities_profiler")


class InstantiationModel(BaseModel):
    steppable_component: PydanticSteppableComponentIFType
    profiler: PydanticSteppableProfilerIFType


@dataclass
class CustomComponentRegisterable:
    component_key: str
    variant_key: str
    custom_component: type
    custom_config: type


class ModalitiesProfilerStarter:
    """Starter class to run profiling either in single process or distributed mode."""

    @staticmethod
    def run_distributed(
        config_file_path: Path,
        experiment_root_path: Path,
        experiment_id: str | None = None,
        custom_component_registerables: list[CustomComponentRegisterable] | None = None,
    ):
        """Run distributed profiling using the Modalities Profiler.
        This method is primarily intended to run large-scale profiling e.g., for model training.

        Args:
            config_file_path (Path): Path to the configuration file.
            experiment_root_path (Path): Root path to store experiment results.
            experiment_id (str, optional): Experiment ID. If None, it will be generated. Defaults to None.
            custom_component_registerables (list[CustomComponentRegisterable], optional): List of custom
                components to register. Defaults to None.
        """
        with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
            if experiment_id is None:
                # get experiment id synched across all ranks
                experiment_id = get_synced_experiment_id_of_run(config_file_path)
            ModalitiesProfilerStarter._copy_config_to_experiment_folder(
                experiment_root_path=experiment_root_path,
                experiment_id=experiment_id,
                config_file_path=config_file_path,
            )

            global_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()

            ModalitiesProfilerStarter._run_helper(
                config_file_path=config_file_path,
                experiment_folder_path=experiment_root_path / experiment_id,
                global_rank=global_rank,
                world_size=world_size,
                custom_component_registerables=custom_component_registerables,
            )

    @staticmethod
    def run_single_process(
        config_file_path: Path,
        experiment_root_path: Path,
        experiment_id: str | None = None,
        custom_component_registerables: list[CustomComponentRegisterable] | None = None,
    ):
        """Run single process profiling using the Modalities Profiler.
        This method is primarily intended for quick profiling experiments
        (e.g., single modules such as custom modules vs Pytorch equivalents) on a single GPU.
        This type of profiling can be seen as a middle ground between distributed profiling and
        single kernel profiling (e.g., using Nsight Compute).

        Args:
            config_file_path (Path): Path to the configuration file.
            experiment_root_path (Path): Root path to store experiment results.
            experiment_id (str, optional): Experiment ID. If None, it will be generated.
            custom_component_registerables (list[CustomComponentRegisterable], optional): List of custom
                components to register. Defaults to None.
        """
        if experiment_id is None:
            # get experiment id synched across all ranks
            experiment_id = get_experiment_id_from_config(config_file_path)

        ModalitiesProfilerStarter._copy_config_to_experiment_folder(
            experiment_root_path=experiment_root_path, experiment_id=experiment_id, config_file_path=config_file_path
        )

        global_rank = 0
        world_size = 1

        ModalitiesProfilerStarter._run_helper(
            config_file_path=config_file_path,
            experiment_folder_path=experiment_root_path / experiment_id,
            global_rank=global_rank,
            world_size=world_size,
            custom_component_registerables=custom_component_registerables,
        )

    @staticmethod
    def _copy_config_to_experiment_folder(
        experiment_root_path: Path, experiment_id: str, config_file_path: Path
    ) -> None:
        # store a copy of the config file in the experiment folder
        if (
            not torch.distributed.is_initialized()  # copy in single process
            or torch.distributed.is_initialized()  # copy only on rank 0 in distributed process
            and torch.distributed.get_rank() == 0
        ):
            experiment_folder_path = experiment_root_path / experiment_id
            experiment_folder_path.mkdir(parents=True, exist_ok=True)
            shutil.copy(config_file_path, experiment_folder_path / config_file_path.name)

    @staticmethod
    def _run_helper(
        config_file_path: Path,
        experiment_folder_path: Path,
        global_rank: int,
        world_size: int,
        custom_component_registerables: list[CustomComponentRegisterable] | None = None,
    ):
        # register custom components and build components from config
        # workaround to avoid triggering synchronization of experiment id in single process
        experiment_id = experiment_folder_path.name if world_size == 1 else None
        main_obj = Main(config_file_path, experiment_id=experiment_id, experiments_root_path=experiment_folder_path)
        if custom_component_registerables is not None:
            for registerable in custom_component_registerables:
                main_obj.add_custom_component(
                    component_key=registerable.component_key,
                    variant_key=registerable.variant_key,
                    custom_component=registerable.custom_component,
                    custom_config=registerable.custom_config,
                )
        components: InstantiationModel = main_obj.build_components(components_model_type=InstantiationModel)
        steppable_component = components.steppable_component
        profiler_cm = components.profiler

        if global_rank == 0:
            step_iterator = trange(len(profiler_cm), desc="Profiling steps")
        else:
            step_iterator = range(len(profiler_cm))

        with profiler_cm as profiler:
            for _ in step_iterator:
                steppable_component.step()
                profiler.step()
