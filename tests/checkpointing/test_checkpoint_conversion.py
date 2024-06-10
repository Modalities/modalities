from pathlib import Path
from modalities.__main__ import _entry_point_convert_pytorch_to_hf_checkpoint


def test_entry_point_convert_pytorch_to_hf_checkpoint():
    checkpoint_dir = Path("/raid/s3/opengptx/alexj/llm_gym/modalities/modalities/data/checkpoints/checkpoint_trained/")
    config_file_name = "config_mem_map_mamba_small_scale.yaml"
    model_file_name = "eid_2024-05-13__08-10-18-model-num_steps_1670000.bin"
    output_hf_checkpoint_dir = checkpoint_dir / "converted_hf_checkpoint"
    test_result = _entry_point_convert_pytorch_to_hf_checkpoint(
        checkpoint_dir,
        config_file_name, 
        model_file_name, 
        output_hf_checkpoint_dir
    )
