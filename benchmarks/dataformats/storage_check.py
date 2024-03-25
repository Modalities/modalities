import os
from pathlib import Path
import sys
from typing import Dict
from rich.console import Console
from rich.table import Table
from benchmarks.dataformats.utils import get_common_opts


def check_storage(paths: Dict[str, Path]):

    results = {}
    for k, p in paths.items():

        if not p:
            continue

        tot_size = 0
        tot_files = 0

        if not os.path.exists(p):
            raise Exception(f"{p} does not exist for path {k}")

        # check if the path is a directory
        if os.path.isdir(p):
            files = os.listdir(p)
        else:
            files = [p]

        # for each file check its size
        for f in files:
            f = os.path.join(p, f)
            tot_size += os.path.getsize(f)
            tot_files += 1

        results[p] = {"total_size": tot_size, "total_files": tot_files}

    # convert the sizes to human readable format
    for p, res in results.items():
        res["total_size"] = f"{res['total_size']/1024/1024:.2f} MB"

    # print the results with rich table
    table = Table(title="Storage check")
    table.add_column("Path", style="cyan", no_wrap=True)
    table.add_column("Total size MB", style="magenta")
    table.add_column("Total files", style="green")
    for p, res in results.items():
        table.add_row(p.__str__(), str(res["total_size"]), str(res["total_files"]))
    console = Console()
    console.print(table)


if __name__ == "__main__":
    opts = get_common_opts(sys.argv[1:])

    # remove file name from webdataset path
    if opts.webdataset:
        opts.webdataset = Path(opts.webdataset).parent

    paths = {
        "webdataset": opts.webdataset,
        "memmap": opts.memmap,
    }

    check_storage(paths)
