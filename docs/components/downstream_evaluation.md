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

**Location:** `src/modalities/evaluator.py` (Lines 205-273)

The `DownstreamEvaluator` simply checks for the existence of an HF checkpoint and launches an evaluation script via a subprocess.

### Behavior
- Triggered if `num_train_steps_done % eval_interval == 0`.
- Only executes on `global_rank == 0`.
- Reads `last_checkpoint_info.json` to find the latest checkpoint.
- Checks if `{checkpoint_path}/hf_checkpoint` exists. If it does NOT exist, evaluation is skipped with a warning (assuming conversion failed or was disabled).
- If the HF checkpoint exists, it formats the `olmes_command_template` and launches it using `subprocess.Popen(cmd, shell=True)`. (It does not wait for it to finish, running it asynchronously).

### Placeholders
The `olmes_command_template` string can use the following placeholders:
- `{hf_model_dir}`: The path to the `{checkpoint_path}/hf_checkpoint` directory.
- `{tasks}`: A comma-separated string of the tasks provided in the config.
- `{step}`: The current `num_train_steps_done`.

### YAML Configuration
```yaml
downstream_evaluator:
  component_key: downstream_evaluator
  variant_key: default
  config:
    tokenizer: ${tokenizer}
    tasks:
      - "arc_challenge::olmes"
      - "hellaswag::olmes"
    eval_interval: 1000
    checkpoint_dir: ${settings.paths.experiments_root_path}/${settings.experiment_id}
    global_rank: ${settings.cuda_env.global_rank}
    olmes_command_template: "olmes --model {hf_model_dir} --tasks {tasks} --output-dir {hf_model_dir}/olmes_eval_{step}"
```

---

## System Integration Summary

For context on how these components are wired into the system, the following files handle the integration:

1. **`src/modalities/trainer.py` (Lines 214-232, 256-268, 435-442)**
   - `conversion_callback` was added to `train()` signature.
   - Pre-loop and in-loop execution order was explicitly set to: `checkpointing_callback` -> `conversion_callback` -> `downstream_evaluation_callback`.

2. **`src/modalities/gym.py` (Lines 45-87)**
   - Threads `conversion_callback` through `Gym.run()` and passes it down to `self.trainer.train()`.

3. **`src/modalities/main.py` (Lines 223-236)**
   - Resolves `components.model_converter.convert` and `components.downstream_evaluator.evaluate`.
   - Passes them into `gym.run()`.

4. **`src/modalities/config/config.py` (Lines 523-535)**
   - Defines Pydantic models `ModelConverterConfig` and `DownstreamEvaluatorConfig`.

5. **`src/modalities/config/instantiation_models.py` (Lines 26, 196-197)**
   - Adds `model_converter` and `downstream_evaluator` fields to `TrainingComponentsInstantiationModel`.

6. **`src/modalities/registry/components.py` (Lines 38, 86, 532)**
   - Registers both classes to the `"default"` component registry.

7. **`tests/test_downstream_evaluator.py`**
   - Contains 11 comprehensive tests mocking the `subprocess` calls and verifying interval gating, rank gating, and directory existence logic.
