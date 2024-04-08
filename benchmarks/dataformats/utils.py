import argparse
import os
from pathlib import Path
from typing import Dict, List



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
