import tempfile
from pathlib import Path

import tqdm
import yaml
from torch.profiler import ProfilerActivity, profile, schedule

from modalities import __main__
from modalities.util import is_launched_via_torchrun, temporary_env
from modalities.utils.profilers.grid_search_utils import GridSearchItem, GridSearchUtils
from modalities.utils.profilers.modalities_profiler import ModalitiesProfiler
from modalities.utils.run_torchrun_script import run_torchrun_with_cleanup


class ModalitiesProfilerStarter:
    @staticmethod
    def run_train_step_profiler(
        config_file_path: Path,
        experiment_folder_path: Path,
        grid_search: list[GridSearchItem],
        num_warmup_steps: int,
        num_measurement_steps: int,
        num_nodes: int = 1,
        node_rank: int = 0,
        rdzv_endpoint: str = "localhost:0",
        rdzv_timeout: int = 30,
        local_rank_ids: list[int] = None,
    ):
        """Applies memory and runtime profiling to the training step of a model training.
        By specifying a grid search, the profiler can be run for multiple configurations.
        Internally, the grid search (i.e., the cartesian product of all settings) is applied
        to the config file and a new temporary config file is created for each grid search item.
        The profiler is then run sequentially for each config file.

        This function can be run in two ways:
        1)  Can be called directly from the command line. In this case, the profiler runs a
            torchrun environment internally that gets destroyed after running each config.
            This makes sure that the profiler is run in a clean environment and that the
            processes are not within an undefined state after OOM errors.
        2)  Can be called from an existing torchrun environment. In this case the grid search
            must contain only a single config, for the same OOM error reasons as above.
            The main purpose for this method is to run or debug a single configuration.
            For a grid search always use the first method.

        Args:
            config_file_path (Path): The path to the config file.
            experiment_folder_path (Path): The path to the experiment folder.
            grid_search (list[GridSearchItem]): The grid search items to be used for the profiler.
            num_warmup_steps (int): The number of warmup steps to be used for the profiler.
                During the warmup steps, the profiler is not measuring the memory and runtime.
            num_measurement_steps (int): The number of measurement steps to be used for the profiler.
                During the measurement steps, the profiler collects the memory and runtime statistics.
            num_nodes (int, optional): The number of nodes to be used. Defaults to 1.
            node_rank (int, optional): The rank of the current node. Defaults to 0.
            rdzv_endpoint: (str, optional): The rendezvous endpoint to be used. Defaults to "localhost:0",
                in which case torchrun selects a free empty port on localhost itself.
            rdzv_timeout: (int, optional): The rendezvous timeout in secons.
            local_rank_ids (list[int], optional): The local rank IDs to be used. Defaults to None.

        Raises:
            RuntimeError: If the profiler is called from a torchrun process with multiple configs.
                The profiler can only be called via torchrun if the grid search has a length of 1.
            RuntimeError: If the profiler is not started from a torchrun or a python process.
        """
        # load the config file
        with open(config_file_path, "r") as f:
            config_string = f.read()
        config_dict = yaml.safe_load(config_string)
        # get one config for each grid search item
        config_dicts = GridSearchUtils.get_configs_from_grid_search(
            config_dict=config_dict,
            grid_search=grid_search,
        )
        # run the profiler for each config
        if len(config_dicts) > 1 and is_launched_via_torchrun():
            raise RuntimeError(
                "TrainStepProfilerStarter.run_train_step_profiler() must not be called via torchrun "
                "with multiple configs (i.e., a grid search). The reason is that recovering from OOM errors "
                "is not possible and the processes need to be killed and restarted. Instead, run this script "
                "without torchrun and modalities will interally start the torchrun processes itself."
            )
        for config_dict in tqdm.tqdm(config_dicts):
            with tempfile.NamedTemporaryFile("w+") as temp_file:
                yaml.dump(config_dict, temp_file)
                temp_file_path = temp_file.name
                if not is_launched_via_torchrun():
                    full_main_path = Path(__main__.__file__).resolve()
                    torch_run_args = [
                        "--nproc_per_node",
                        str(len(local_rank_ids)),
                        "--nnodes",
                        str(num_nodes),
                        "--node_rank",
                        str(node_rank),
                        "--rdzv_backend",
                        "c10d",
                        "--rdzv_endpoint",
                        rdzv_endpoint,
                        "--rdzv_timeout",
                        rdzv_timeout,
                    ]
                    modalities_args = [
                        str(full_main_path),
                        "profile",
                        "train_step",
                        "--config_file_path",
                        str(temp_file_path),
                        "--experiment_folder_path",
                        str(experiment_folder_path),
                        "--num_measurement_steps",
                        str(num_measurement_steps),
                        "--num_warmup_steps",
                        str(num_warmup_steps),
                    ]
                    if local_rank_ids is not None:
                        env_vars = {"CUDA_VISIBLE_DEVICES": ",".join(map(str, local_rank_ids))}
                    else:
                        env_vars = {}
                    with temporary_env(env_vars):
                        run_torchrun_with_cleanup(torch_run_args=torch_run_args, script_args=modalities_args)
                else:
                    ModalitiesProfiler.get_train_step_statistics(
                        config_file_path=temp_file_path,
                        num_warmup_steps=num_warmup_steps,
                        num_measurement_steps=num_measurement_steps,
                        experiment_folder_path=experiment_folder_path,
                    )

    @staticmethod
    def get_forward_pass_profiling(
        num_measurements: int, config_file_path: Path, profiler_activities: list[ProfilerActivity] = None
    ) -> profile:
        if profiler_activities is None:
            profiler_activities = [ProfilerActivity.CUDA]

        profiler_context_manager = profile(
            activities=profiler_activities,
            schedule=schedule(wait=2, warmup=2, active=num_measurements),
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
            with_stack=True,
            with_modules=True,
        )
        ModalitiesProfiler.get_forward_pass_profiling(
            config_file_path=config_file_path,
            num_measurement_steps=num_measurements,
            profile_context_manager=profiler_context_manager,
        )
        return profiler_context_manager
