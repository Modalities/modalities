import argparse
import copy
import os

from datasets import load_dataset
from oe_eval.configs.tasks import TASK_CONFIGS
from oe_eval.launch import resolve_task_suite
from oe_eval.run_eval import load_task


def main():
    parser = argparse.ArgumentParser(description="Precache OLMES tasks and required HF datasets.")
    parser.add_argument(
        "--tasks", 
        nargs="+", 
        required=True, 
        help="List of OLMES tasks to precache (e.g. arc_challenge:rc::olmes:full hellaswag:rc::olmes:full)"
    )
    args = parser.parse_args()

    hf_home = os.environ.get("HF_DATASETS_CACHE", os.environ.get("HF_HOME", "~/.cache/huggingface"))
    print(f"HF_DATASETS_CACHE is set to: {hf_home}")

    # ---- Part 1: OLMES tasks ----
    print("\n--- Caching OLMES tasks ---")
    all_tasks = []
    for t in args.tasks:
        try:
            all_tasks += resolve_task_suite(t, {})
        except Exception as e:
            print(f"!! could not resolve {t}: {e}")

    print(f"\nWill download {len(all_tasks)} tasks to {hf_home}")
    for task_name in all_tasks:
        if task_name not in TASK_CONFIGS:
            print(f"?? not in TASK_CONFIGS: {task_name}")
            continue
        try:
            cfg = copy.deepcopy(TASK_CONFIGS[task_name])
            task = load_task(cfg, ".")
            print(f"-> downloading {task_name}")
            task.download()
            print(f"   done")
        except Exception as e:
            print(f"!! failed {task_name}: {e}")

    # ---- Part 2: pseudo-sources used by paloma_diagnostics.py ----
    print("\n--- Caching diagnostics pseudo-source datasets (gsm8k, trivia_qa) ---")
    specs = [
        ('gsm8k',     'main',         'test'),
        ('trivia_qa', 'rc.nocontext', 'validation'),
    ]
    for path, config, split in specs:
        try:
            print(f"-> downloading {path} ({config}, {split})")
            load_dataset(path, config, split=split)
            print(f"   done")
        except Exception as e:
            print(f"!! failed {path}: {e}")

if __name__ == "__main__":
    main()
