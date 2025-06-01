import argparse
import os
from pathlib import Path

from modalities.utils.profilers.grid_search_utils import GridSearchItem
from modalities.utils.profilers.profiler_starters import ModalitiesProfilerStarter


def main():
    parser = argparse.ArgumentParser(description="Run a training step profiler with grid search.")

    parser.add_argument(
        "--config_file", type=str, default="config_8B_scaling.yaml", help="Path to the configuration YAML file."
    )
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes to use.")
    parser.add_argument("--num_warmup_steps", type=int, default=16, help="Number of warmup steps before measurement.")
    parser.add_argument("--num_measurement_steps", type=int, default=16, help="Number of measurement steps.")
    parser.add_argument(
        "--rdzv_endpoint", type=str, default="localhost:0", help="Rendezvous endpoint for distributed training."
    )
    parser.add_argument("--rdzv_timeout", type=int, default=16, help="Rendezvous timeout in seconds.")
    parser.add_argument(
        "--experiment_folder",
        type=str,
        default="experiments",
        help="Path to the folder where experiments will be stored.",
    )

    args = parser.parse_args()

    # Read the environment variable
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible_devices != "":
        local_rank_ids = [int(dev.strip()) for dev in cuda_visible_devices.split(",")]
        if len(local_rank_ids) == 0:
            raise ValueError(f"No local ranks specified in CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
    else:
        raise ValueError("CUDA_VISIBLE_DEVICES must be set.")

    print(f"Running script on {args.num_nodes} nodes with the following arguments:")
    print(args)

    config_file_path = Path(args.config_file).resolve()

    experiment_folder_path = Path(args.experiment_folder).resolve()
    experiment_folder_path = experiment_folder_path / f"num_ranks_{len(local_rank_ids)*args.num_nodes}"
    experiment_folder_path.mkdir(exist_ok=True)

    grid_search = [
        GridSearchItem(name="settings.benchmark.batch_size", values=list(range(1, 4))),
        GridSearchItem(name="settings.benchmark.sequence_length", values=[2048, 4096, 8192]),
        GridSearchItem(name="settings.benchmark.vocab_size", values=[50304]),
    ]

    ModalitiesProfilerStarter.run_train_step_profiler(
        config_file_path=config_file_path,
        experiment_folder_path=experiment_folder_path,
        grid_search=grid_search,
        num_warmup_steps=args.num_warmup_steps,
        num_measurement_steps=args.num_measurement_steps,
        num_nodes=args.num_nodes,
        rdzv_endpoint=args.rdzv_endpoint,
        rdzv_timeout=args.rdzv_timeout,
        local_rank_ids=local_rank_ids,
    )


if __name__ == "__main__":
    main()
