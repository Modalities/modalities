checkpoint_dir_path="checkpoints/2025-07-02__17-25-05_91a54afb/eid_2025-07-02__17-25-05_91a54afb-seen_steps_438-seen_tokens_43057152-target_steps_584-target_tokens_57409536"
converted_checkpoint_file_path="$checkpoint_dir_path/converted_checkpoint.pth"
python -m torch.distributed.checkpoint.format_utils dcp_to_torch  $checkpoint_dir_path $converted_checkpoint_file_path
