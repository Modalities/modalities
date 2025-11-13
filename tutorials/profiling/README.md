## Profiling Tutorial

This tutorial shows how to profile both single components (e.g. a normalization layer) and full distributed model runs using the Modalities profiling utilities. It provides ready-to-run scripts plus configuration files that control model, data, and profiler behavior. Output artifacts (HTML, JSON, TXT) capture performance, trace timelines, and memory usage for analysis.

---

### Directory Layout

```
tutorials/profiling/
  configs/
    distributed_8B_model_profiling.yaml         # Config for multi‑GPU (8B model) distributed run
    single_process_rms_norm_profiling.yaml      # Config used by the RMS norm single‑process example
    small_profiling_config.yaml                 # Minimal / smaller test configuration
  scripts/
    single_process/
      single_process_norm_profiling.py          # Python entrypoint for profiling a single norm component
      single_process_profiler_starter.sh        # Convenience shell wrapper
    distributed/
      run_distributed_model_profiling.py        # Python entrypoint for distributed profiling
      distributed_profiler_starter.sh           # Torchrun launcher
  experiments/
    <timestamp>_<hash>/                         # One folder per profiling run; contains reports & a copy of the config
      profiler_summary_ranks_<N>_rank_<r>.txt   # Text summary
      profiler_trace_ranks_<N>_rank_<r>.json    # Chrome trace JSON (timeline)
      profiler_memory_ranks_<N>_rank_<r>.html   # Memory usage HTML report
      <original_config>.yaml                    # Snapshot of the config used for reproducibility
```

Each run produces a uniquely named experiment directory: `YYYY-MM-DD__HH-MM-SS_<short_hash>/`.

---

### Core Concepts

| Concept | Meaning |
|---------|---------|
| `warmup_steps` | Steps ignored for measurement; allows kernels & caches to stabilize. |
| `wait_steps` | Idle or light steps between measurements to reduce interference. |
| `num_measurement_steps` | Number of timed/profiled iterations collected after warmup. |
| `profiled_ranks` (distributed) | Subset of ranks for which detailed profiler artifacts are saved (others run normally). |
| Custom Component | A user-defined steppable unit (implements `SteppableComponentIF.step()`) inserted into single-process profiling flows. |

---

### Single-Process Component Profiling

Entry script: `scripts/single_process/single_process_norm_profiling.py`

This defines a custom steppable component `SteppableNorm` that:
1. Receives a dataset batch generator and a `norm` module from the config.
2. Moves the module to GPU (`cuda`) and casts to `bfloat16`.
3. Optionally applies `torch.compile` for compiler optimizations (`apply_compile=True`).
4. In each `step()`, obtains a batch via `get_dataset_batch()`, moves it to device, and records a profiler region `rms_norm_inference` while invoking `self.norm(batch.samples['input_ids'])`.

Registration is done via a `CustomComponentRegisterable` so the profiler starter can instantiate the component from YAML-defined parameters.

Run options:

```sh
# From tutorials/profiling/scripts/single_process/
sh single_process_profiler_starter.sh
```

Adjust measurement parameters inside the script (edit `num_measurements`, `wait`, `warmup`) or make them configurable via CLI if desired.

Artifacts land in `tutorials/profiling/experiments/<timestamp_hash>/`.

#### Adding Another Component

For using a custom steppable component add the component to the registry as follows: 

1. Implement a new class extending `SteppableComponentIF` with a `step()` method.
2. Create a corresponding `Config` (Pydantic model) listing required fields (e.g. a model variant, dataset generator, flags).
3. Add a `CustomComponentRegisterable` entry to the `custom_component_registerables` list.
4. Reference the new `variant_key` within your YAML config under the expected component key.

Note, other components (e.g., a custom norm implementation) can be added in the same fashion and used by adding them to the YAML configuration and wiring them up with the other components.  

---

### Distributed Profiling

Entry scripts:
* `scripts/distributed/run_distributed_model_profiling.py` – calls `ModalitiesProfilerStarter.run_distributed(...)`.
* `scripts/distributed/distributed_profiler_starter.sh` – launches via `torchrun`.

Important parameters shown in the script:
```python
num_measurements = 3
wait = 20
warmup = 20
profiled_ranks = [0, 1]
```
Only ranks listed in `profiled_ranks` emit full trace/memory artifacts, reducing overhead when scaling.

Launch example:
```sh
cd tutorials/profiling/scripts/distributed/
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
sh distributed_profiler_starter.sh
```

Key `torchrun` flags in the starter script:
* `--nproc_per_node 8` – number of processes (match GPUs visible)
* `--rdzv-endpoint localhost:29589` – rendezvous for process group formation

Tune `CUDA_VISIBLE_DEVICES` and `--nproc_per_node` to match available hardware. Update the YAML (`configs/distributed_8B_model_profiling.yaml`) for model scale, dataset, batch sizes, or sharding parameters.

---

### Understanding Output Artifacts

| File | Use |
|------|-----|
| `profiler_summary_*.txt` | Aggregated statistics per measurement step: timings, averages, quantiles. Start here for quick comparisons. |
| `profiler_trace_*.json` | Chrome trace format; load at `chrome://tracing` or Perfetto to inspect timeline, kernel durations, overlaps. |
| `profiler_memory_*.html` | Memory trend visualization (allocated, reserved, peak). Useful for spotting fragmentation or spikes. |
| `<config>.yaml` | Exact snapshot of configuration enabling reproducibility & diffing across runs. |

---

### Quick Start Summary

Single process RMS Norm:
```bash
cd tutorials/profiling/scripts/single_process/
python single_process_norm_profiling.py
```

Distributed model profiling (8 GPUs example):
```bash
cd tutorials/profiling/scripts/distributed/
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --rdzv-endpoint localhost:29589 --nnodes 1 --nproc_per_node 8 run_distributed_model_profiling.py
```

Locate results:
```bash
ls ../../experiments/*
```

Load trace in Chrome:
1. Copy `profiler_trace_ranks_1_rank_0.json` locally.
2. Open `chrome://tracing` → Load.
