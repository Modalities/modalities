from pathlib import Path

from modalities.util import get_experiment_id_from_config
from modalities.utils.profilers.grid_search_utils import GridSearchItem
from modalities.utils.profilers.profiler_starters import TrainStepProfilerStarter

if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    config_file_path = current_dir / "config_activation_checkpointing_fsdp2_benchmark_8B.yaml"

    experiment_folder_path = current_dir / "experiments"
    experiment_id = get_experiment_id_from_config(config_file_path)

    grid_search = [
        GridSearchItem(name="settings.benchmark.batch_size", values=list(range(1, 5))),
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
        nproc_per_node=8,
        num_nodes=1,
        rdzv_endpoint="localhost:0",
    )
