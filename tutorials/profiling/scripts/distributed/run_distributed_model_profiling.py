from pathlib import Path

from modalities.utils.profilers.modalities_profiler import ModalitiesProfilerStarter

if __name__ == "__main__":
    cwd = Path(__file__).parent.resolve()
    config_path = cwd / Path("../../configs/distributed_8B_model_profiling.yaml")
    experiment_root_path = cwd / Path("../../experiments/")

    ModalitiesProfilerStarter.run_distributed(
        config_file_path=config_path,
        experiment_root_path=experiment_root_path,
    )
