# Downstream Evaluation Pipeline

## Overview

The downstream evaluation pipeline in Modalities is a decoupled, three-stage callback system that executes at configurable step intervals during the training loop.

The order of execution inside `Trainer.train` is:
1. `checkpointing_callback`: Saves the PyTorch/FSDP checkpoint to disk.
2. `conversion_callback`: (Optional) Converts the PyTorch checkpoint to a Hugging Face (HF) checkpoint.
3. `downstream_evaluation_callback`: (Optional) Runs external evaluation tools (like OLMES) on the newly created HF checkpoint.

By keeping conversion and evaluation decoupled, you can configure just the converter, just the evaluator (if HF checkpoints are generated elsewhere), or both.

---

## 1. Conversion Callback (`ModelConverter`)

**Location:** `src/modalities/conversion/model_converter.py` (Lines 10-67)

The `ModelConverter` is a thin wrapper that executes a shell command template via a subprocess.

### Behavior
- Triggered if `num_train_steps_done % eval_interval == 0`.
- Only executes on `global_rank == 0`.
- Reads `last_checkpoint_info.json` from the checkpoint directory to determine the latest checkpoint path.
- Checks if the `{checkpoint_path}/hf_checkpoint` directory already exists. If it does, conversion is skipped.
- If it does not exist, it formats the `command_template` and runs it using `subprocess.run(cmd, shell=True, check=True)`.

### Placeholders
The `command_template` string can use the following placeholders:
- `{checkpoint_path}`: The path to the latest checkpoint directory (resolved at runtime).
- `{output_dir}`: Evaluates to `{checkpoint_path}/hf_checkpoint`.
- `{modalities_config}`: Path to the YAML config file found inside or next to the checkpoint directory.

### YAML Configuration
```yaml
model_converter:
  component_key: model_converter
  variant_key: default
  config:
    command_template: "python src/modalities/conversion/gpt2/convert_gpt2.py {modalities_config} {output_dir} --checkpoint_path {checkpoint_path}"
    checkpoint_dir: ${settings.paths.experiments_root_path}/${settings.experiment_id}
    global_rank: ${settings.cuda_env.global_rank}
    eval_interval: 1000
```

---

## 2. Downstream Evaluation Callback (`DownstreamEvaluator`)

**Location:** `src/modalities/evaluator.py` (Lines 210-335)

The `DownstreamEvaluator` checks for the existence of an HF checkpoint, launches an evaluation script via a subprocess, tracks active processes, and syncs OLMES metrics to the active W&B run.

### Behavior
- Triggered if `num_train_steps_done % eval_interval == 0`.
- Only executes on `global_rank == 0`.
- Reads `last_checkpoint_info.json` to find the latest checkpoint.
- Checks if `{checkpoint_path}/hf_checkpoint` exists. If it does NOT exist, evaluation is skipped with a warning (assuming conversion failed or was disabled).
- If the HF checkpoint exists, it formats the `olmes_command_template` and launches it asynchronously using `subprocess.Popen(cmd, shell=True)`.
- **Process Tracking**: Stores `(Popen, step, hf_model_dir)` tuples in `self.active_processes` (Lines 233, 258).
- **Graceful Exit**: `wait_for_evaluations()` (Lines 264-275) iterates over `active_processes`, calls `.wait()`, and syncs metrics after each evaluation completes.
- **W&B Metric Sync**: `_sync_metrics_to_wandb()` (Lines 277-315) parses `metrics-all.jsonl` from the OLMES output directory, extracts `primary_score` for each task alias, and logs them to the active `wandb.run` as `eval/{alias}` at the correct training step. Gracefully skips if W&B is disabled or not installed.

### Placeholders
The `olmes_command_template` string can use the following placeholders:
- `{hf_model_dir}`: The path to the `{checkpoint_path}/hf_checkpoint` directory.
- `{tasks}`: A space-separated string of the tasks provided in the config (Line 248).
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
    olmes_command_template: "CUDA_VISIBLE_DEVICES=$LOCAL_RANK . /home/markus_frey/Github/olmes/.venv/bin/activate && olmes --model {hf_model_dir} --model-args '{{\"trust_remote_code\": true}}' --task {tasks} --limit 128 --output-dir {hf_model_dir}/olmes_eval_{step} > {hf_model_dir}/olmes_eval_{step}.log 2>&1"
```

---

## System Integration Summary

For context on how these components are wired into the system, the following files handle the integration:

1. **`src/modalities/trainer.py`**
   - `conversion_callback` was added to `train()` signature.
   - Pre-loop and in-loop execution order was explicitly set to: `checkpointing_callback` -> `conversion_callback` -> `downstream_evaluation_callback`.

2. **`src/modalities/gym.py`**
   - Threads `conversion_callback` through `Gym.run()` and passes it down to `self.trainer.train()`.

3. **`src/modalities/main.py` (Lines 227-249)**
   - Resolves `components.model_converter.convert` and `components.downstream_evaluator.evaluate`.
   - Passes them into `gym.run()`.
   - **Post-Training Wait** (Lines 244-249): At the very end of `run()`, explicitly calls `components.downstream_evaluator.wait_for_evaluations()` with prominent `print_rank_0` logging to ensure training does not exit until evaluations complete.

4. **`src/modalities/config/config.py`**
   - Defines Pydantic models `ModelConverterConfig` and `DownstreamEvaluatorConfig`.

5. **`src/modalities/config/instantiation_models.py`**
   - Adds `model_converter` and `downstream_evaluator` fields to `TrainingComponentsInstantiationModel`.

6. **`src/modalities/registry/components.py`**
   - Registers both classes to the `"default"` component registry.

7. **`src/modalities/conversion/gpt2/convert_gpt2.py` (Lines 105-112)**
   - Updated to support Hugging Face tokenizers (`pretrained_hf_tokenizer`) alongside SentencePiece. Detects tokenizer configs and saves `vocab.json` / `tokenizer.json` directly to the `hf_checkpoint` directory.

8. **`tests/test_downstream_evaluator.py`**
   - Contains comprehensive tests mocking the `subprocess` calls and verifying interval gating, rank gating, and directory existence logic.
