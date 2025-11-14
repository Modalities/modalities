from pathlib import Path

from modalities.utils.profilers.modalities_profiler import ModalitiesProfilerStarter

if __name__ == "__main__":
    cwd = Path(__file__).parent.resolve()
    config_path = cwd / Path("../../configs/distributed_8B_model_profiling.yaml")
    experiment_root_path = Path("../../experiments/")

    num_measurement_steps = 3
    num_wait_steps = 20
    num_warmup_steps = 20
    profiled_ranks = [0, 1]

    ModalitiesProfilerStarter.run_distributed(
        config_file_path=config_path,
        num_measurement_steps=num_measurement_steps,
        num_wait_steps=num_wait_steps,
        num_warmup_steps=num_warmup_steps,
        experiment_root_path=experiment_root_path,
        profiled_ranks=profiled_ranks,
    )
