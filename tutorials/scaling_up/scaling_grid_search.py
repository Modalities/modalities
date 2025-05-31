from pathlib import Path

from modalities.util import get_experiment_id_from_config
from modalities.utils.profilers.grid_search_utils import GridSearchItem
from modalities.utils.profilers.profiler_starters import ModalitiesProfilerStarter

if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    config_file_path = current_dir / "config_8B_scaling.yaml"

    experiment_folder_path = current_dir / "experiments"
    experiment_id = get_experiment_id_from_config(config_file_path)

    grid_search = [
        GridSearchItem(name="settings.benchmark.batch_size", values=list(range(1, 64))),
        GridSearchItem(name="settings.benchmark.sequence_length", values=[128, 256, 512, 1024, 2048, 4096, 8192]),
        GridSearchItem(name="settings.benchmark.vocab_size", values=[50304]),
    ]
    ModalitiesProfilerStarter.run_train_step_profiler(
        config_file_path=config_file_path,
        experiment_folder_path=experiment_folder_path / experiment_id,
        grid_search=grid_search,
        num_warmup_steps=16,
        num_measurement_steps=16,
        nproc_per_node=8,
        num_nodes=1,
        rdzv_endpoint="localhost:0",
        local_rank_ids=[0, 1, 2, 3, 4, 5, 6, 7],
    )
