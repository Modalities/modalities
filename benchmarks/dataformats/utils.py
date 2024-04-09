import argparse
import os
from pathlib import Path
import time
from memory_profiler import memory_usage
import numpy as np


def profile(runs=1):
    """Decorator to profile time and memory usage of a function, running it multiple times."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            times = []
            mem_usages = []
            for _ in range(runs):
                start_time = time.time()
                mem_before = memory_usage(interval=0.1, timeout=1)
                result = func(*args, **kwargs)
                mem_after = memory_usage(interval=0.1, timeout=1)
                times.append(time.time() - start_time)
                mem_usages.append(max(mem_after) - min(mem_before))
            avg_time = np.mean(times)
            std_time = np.std(times)
            avg_mem = np.mean(mem_usages)
            std_mem = np.std(mem_usages)
            print(f"Function {func.__name__} on average took {avg_time:.2f}s (±{std_time:.2f}) and used {avg_mem:.2f} MiB (±{std_mem:.2f}) over {runs} runs")
            return result
        return wrapper
    return decorator


def check_web_data(path2dir: Path) -> str:

    if not path2dir.exists():
            raise Exception("Webdataset path not found")
    

    # get the list of files in the directory
    files = os.listdir(path2dir)
    # get only tar files
    files = [f for f in files if f.endswith(".tar")]

    if len(files) == 0:
        raise Exception("No tar files found in the directory")


    first_file = files[0].split(".")[0]
    last_file = files[-1].split(".")[0]

    # append {000000..00000X}.tar to the path
    ext = f"{path2dir}/{{{first_file}..{last_file}}}.tar"
    return ext


def check_mmapped_data(memmap_path: Path) -> None:
   
    if not memmap_path.exists():
        raise Exception("Memmap path not found")
    
    # assert the extension is pbin
    if memmap_path.suffix != ".pbin":
        raise Exception("Memmap file extension should be .pbin")
    


def get_common_opts(params) -> argparse.Namespace:
    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--json_file",
        type=str,
        default="",
        help="Path to a jsonL file from karpathy split",
    )


    parser.add_argument(
        "--webout",
        type=str,
        default="data/",
        help="Path to the directory where the webdataset will be created",
    )

    parser.add_argument(
        "--webdata",
        type=str,
        default="",
        help="Path to the webdataset directory",
    )

    parser.add_argument(
        "--memmap",
        type=str,
        default="",
        help="Path to the memmap directory",
    )


    parser.add_argument(
        "--no_iter",
        action="store_true",
        help="Do not run iteration benchmarks",
    )

    opts=parser.parse_args(params)


    if opts.webdata:
        opts.webdata = Path(opts.webdata)
        opts.webdata = check_web_data(opts.webdata)
        
    if opts.memmap:
        opts.memmap = Path(opts.memmap)
        check_mmapped_data(opts.memmap)


    return opts
