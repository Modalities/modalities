from pathlib import Path

from modalities.util import get_experiment_id_from_config
from modalities.utils.profilers.grid_search_utils import GridSearchItem
from modalities.utils.profilers.profiler_starters import ModalitiesProfilerStarter

if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    config_file_path = current_dir / "config_activation_checkpointing_fsdp2_benchmark_8B.yaml"

    experiment_folder_path = current_dir / "experiments"
    experiment_id = get_experiment_id_from_config(config_file_path)

    grid_search = [
        GridSearchItem(name="settings.benchmark.batch_size", values=list(range(1, 10))),
        GridSearchItem(name="settings.benchmark.sequence_length", values=[4096]),
        GridSearchItem(name="settings.benchmark.vocab_size", values=[50304]),
        GridSearchItem(
            name="settings.benchmark.ac_ops_keys",
            values=[
                [],
                ["torch.ops.aten.mm.default"],
                ["torch.ops.aten._scaled_dot_product_efficient_attention.default"],
                ["torch.ops.aten._scaled_dot_product_flash_attention.default"],
                ["torch.ops._c10d_functional.reduce_scatter_tensor.default"],
                ["torch.ops.aten.max.default"],
            ],
        ),
    ]
    ModalitiesProfilerStarter.run_train_step_profiler(
        config_file_path=config_file_path,
        experiment_folder_path=experiment_folder_path / experiment_id,
        grid_search=grid_search,
        num_warmup_steps=2,
        num_measurement_steps=5,
        nproc_per_node=8,
        num_nodes=1,
        rdzv_endpoint="localhost:0",
    )
