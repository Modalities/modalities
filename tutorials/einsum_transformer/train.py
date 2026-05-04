import os
from pathlib import Path

from model.einsum_transformer import EinsumTransformer
from pydantic import BaseModel

from modalities.__main__ import Main
from modalities.config.config import ProcessGroupBackendType
from modalities.config.instantiation_models import TrainingComponentsInstantiationModel
from modalities.running_env.cuda_env import CudaEnv


class EinsumTransformerConfig(BaseModel):
    prediction_key: str
    sample_key: str
    vocab_size: int
    sequence_length: int
    embed_dim: int
    num_q_heads: int
    num_kv_heads: int
    num_layers: int
    mlp_expansion_factor: float = 4


def main():
    # load and parse the config file
    cwd = Path(__file__).parent
    # change to cwd
    os.chdir(cwd)
    config_file_path = cwd / Path("einsum_transformer_config.yaml")
    experiments_root_path = cwd / Path("experiments")

    with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
        # instantiate the Main entrypoint of modalities by passing in the config path
        modalities_main = Main(config_path=config_file_path, experiments_root_path=experiments_root_path)

        # add the custom component to modalities
        modalities_main.add_custom_component(
            component_key="model",
            variant_key="einsum_transformer",
            custom_component=EinsumTransformer,
            custom_config=EinsumTransformerConfig,
        )
        # run the experiment
        components: TrainingComponentsInstantiationModel = modalities_main.build_components(
            components_model_type=TrainingComponentsInstantiationModel
        )
        modalities_main.run(components)


if __name__ == "__main__":
    main()
