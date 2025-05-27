import os
from pathlib import Path

from torch.profiler import ProfilerActivity

from modalities.utils.profilers.profiler_starters import ModalitiesProfilerStarter

if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    config_file_path = current_dir / "config_activation_checkpointing_fsdp2_benchmark_small.yaml"

    num_measurements = 10
    profiler = ModalitiesProfilerStarter.get_forward_pass_profiling(
        num_measurements=num_measurements,
        config_file_path=config_file_path,
        profiler_activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    )

    print(
        profiler.key_averages().table(
            sort_by="self_cuda_time_total",
            # row_limit=row_limit,
            max_name_column_width=240,
        )
    )
    if int(os.environ["RANK"]) == 0:
        profiler.export_chrome_trace(os.path.join(config_file_path.parent, "activation_checkpointing_profile.json"))

    pass


#     # Now, access recorded function events
# # 3. AFTER the "with" closes, THEN inspect!

# if int(os.environ["RANK"]) == 0:
#     events = profiler.events()   # <-- only now!

#     aten_to_module = []

#     for evt in events:
#         if evt.name.startswith("aten::"):
#             module = None
#             if evt.stack:
#                 for frame in evt.stack:
#                     if 'forward' in frame.name:
#                         module = frame.name
#                         break
#             aten_to_module.append((evt.name, module))

#     print(f"Found {len(aten_to_module)} ATen ops!")
#     for aten, module in aten_to_module:
#         if module is not None:
#             print(f"ATen: {aten:30s} --> Module: {module}")

# pass
