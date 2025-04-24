import tempfile
from pathlib import Path

import tqdm
import yaml

from modalities import __main__
from modalities.util import get_experiment_id_from_config, is_launched_via_torchrun
from modalities.utils.profilers.grid_search_utils import GridSearchItem, GridSearchUtils
from modalities.utils.profilers.modalities_profiler import ModalitiesProfiler
from modalities.utils.run_torchrun_script import run_torchrun_with_cleanup


class TrainStepProfilerStarter:
    @staticmethod
    def run_train_step_profiler(
        config_file_path: Path,
        experiment_folder_path: Path,
        grid_search: list[GridSearchItem],
        num_warmup_steps: int,
        num_measurement_steps: int,
        num_ranks: int,
        num_nodes: int = 1,
        debug: bool = False,
    ):
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
        for config_dict in tqdm.tqdm(config_dicts):
            with tempfile.NamedTemporaryFile("w+") as temp_file:
                yaml.dump(config_dict, temp_file)
                temp_file_path = temp_file.name
                # TODO call subprocdess here with torchrun command
                if not is_launched_via_torchrun():
                    full_main_path = Path(__main__.__file__).resolve()
                    torch_run_args = [
                        "--nproc_per_node",
                        str(num_ranks),
                        "--nnodes",
                        str(num_nodes),
                        "--rdzv_id",
                        "0",
                        "--rdzv_backend",
                        "c10d",
                        "--rdzv_endpoint",
                        "localhost:0",
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
                    run_torchrun_with_cleanup(torch_run_args=torch_run_args, script_args=modalities_args)
                elif is_launched_via_torchrun() and debug:
                    ModalitiesProfiler.get_train_step_statistics(
                        config_file_path=temp_file_path,
                        num_warmup_steps=num_warmup_steps,
                        num_measurement_steps=num_measurement_steps,
                        experiment_folder_path=experiment_folder_path,
                    )
                else:
                    raise RuntimeError(
                        "TrainStepProfilerStarter.run_train_step_profiler() must not be called from a torchrun process."
                    )


if __name__ == "__main__":
    config_file_path = Path(
        "/raid/s3/opengptx/max_lue/repositories/modalities/tests/training/config_activation_checkpointing_fsdp2_benchmark_8B.yaml"
    )

    experiment_folder_path = Path("/raid/s3/opengptx/max_lue/repositories/modalities/tests/training/benchmark")
    experiment_id = get_experiment_id_from_config(config_file_path)

    grid_search = [
        GridSearchItem(name="settings.benchmark.batch_size", values=list(range(1, 25))),
        GridSearchItem(name="settings.benchmark.sequence_length", values=[4096]),
        GridSearchItem(name="settings.benchmark.vocab_size", values=[50304]),
        GridSearchItem(
            name="settings.benchmark.ac_mode",
            values=[
                "model_raw",
                "full_activation_checkpointed_model",
                "selective_layer_activation_checkpointed_model",
                "selective_op_activation_checkpointed_model",
            ],
        ),
    ]
    TrainStepProfilerStarter.run_train_step_profiler(
        config_file_path=config_file_path,
        experiment_folder_path=experiment_folder_path / experiment_id,
        grid_search=grid_search,
        num_warmup_steps=2,
        num_measurement_steps=5,
        num_ranks=8,
        num_nodes=1,
    )
