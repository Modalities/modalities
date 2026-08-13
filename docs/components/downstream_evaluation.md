# Downstream Evaluation Pipeline

The downstream evaluation pipeline in Modalities is a callback system that executes at configurable step intervals during the training loop.

The order of execution inside `Trainer.train` is:
1. `checkpointing_callback`: Saves the PyTorch/FSDP checkpoint to disk.
2. `downstream_evaluation_callback`: (Optional) Runs external evaluation tools (like OLMES) on the newly created HF checkpoint.

---

## Downstream Evaluation Callback (`DownstreamEvaluator`)

**Location:** `src/modalities/evaluator.py`

The `DownstreamEvaluator` checks for the existence of an HF checkpoint, launches an evaluation script via a subprocess, tracks active processes, and syncs OLMES metrics to the active W&B run.

### Behavior
- Triggered if `num_train_steps_done % eval_interval == 0`.
- Only executes on `global_rank == 0`.
- Reads `last_checkpoint_info.json` to find the latest checkpoint.
- Checks if `{checkpoint_path}/hf_checkpoint` exists. If it does NOT exist, evaluation is skipped with a warning (assuming conversion failed or was disabled).
- If the HF checkpoint exists, it formats the `olmes_command_template` and launches it asynchronously using `subprocess.Popen(cmd, shell=True)`.
- **Process Tracking**: Stores `(Popen, step, hf_model_dir)` tuples in `self.active_processes`.
- **Graceful Exit**: `wait_for_evaluations()` iterates over `active_processes`, calls `.wait()`, and syncs metrics after each evaluation completes.
- **W&B Metric Sync**: Extracts `primary_score` for each task alias from the OLMES output file and logs them to the active `wandb.run` as `downstream/{alias}` at the correct training step. Gracefully skips if W&B is disabled or not installed.

### Placeholders
The `olmes_command_template` string can use the following placeholders:
- `{hf_model_dir}`: The path to the `{checkpoint_path}/hf_checkpoint` directory.
- `{tasks}`: A space-separated string of the tasks provided in the config.
- `{step}`: The current `num_train_steps_done`.

### YAML Configuration
```yaml
downstream_evaluator:
  component_key: downstream_evaluator
  variant_key: default
  config:
    tokenizer:
      instance_key: tokenizer
      pass_type: BY_REFERENCE
    tasks:
      - "arc_challenge::olmes"
      - "hellaswag::olmes"
    eval_interval: 100
    checkpoint_dir: ${settings.paths.experiments_root_path}/${settings.experiment_id}
    global_rank: ${settings.cuda_env.global_rank}
    olmes_command_template: "bash scripts/evaluation/run_olmes_sbatch.sh {hf_model_dir} '{tasks}' {step} 1024 1"
```
