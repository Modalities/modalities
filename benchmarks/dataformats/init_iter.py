import gc
import sys
from pathlib import Path
import time
from memory_profiler import memory_usage

from benchmarks.dataformats.utils import get_common_opts, profile
import webdataset as wds
from src.modalities.dataloader.dataset import PackedMemMapDataset

@profile(runs=5)
def initialization_memmap(data_path: Path) -> PackedMemMapDataset:
    sample_key = ["img_path", "text0", "text1", "text2", "text3", "text4"]
    dataset = PackedMemMapDataset(raw_data_path=data_path, sample_keys=sample_key)
    return dataset

@profile(runs=5)
def initialization_webdataset(data_path: Path) -> wds.WebDataset:
    pil_dataset = wds.WebDataset(data_path)
    return pil_dataset

@profile(runs=2)
def memmap_iteration(dataset: PackedMemMapDataset):
    count = 0
    for i in range(len(dataset)):
        p = dataset[i]
        count += 1
    print(f"Total images: {count}")

@profile(runs=2)
def iteration_webdataset(web_data: wds.WebDataset):
    web_data = web_data.decode("pil").to_tuple("jpg", "json")
    count = 0
    for image, json in web_data:
        count += 1
    print(f"Total images: {count}")

if __name__ == "__main__":
    opts = get_common_opts(sys.argv[1:])
    
    if opts.webdata:
        data_path = opts.webdata
        print(f"Web data path: {data_path}")
        web_data = initialization_webdataset(data_path)
        print("Webdataset initialized")
        
        if not opts.no_iter:
            print("\nIterating over the dataset...")
            iteration_webdataset(web_data)
            print("Webdataset iteration completed")
        
        # Collect garbage
        del web_data
        gc.collect()

    if opts.memmap:
        print(f"Memmap data path: {opts.memmap}")
        memmap_data = initialization_memmap(opts.memmap)
        print("Memmap initialized")

        if not opts.no_iter:
            print("\nIterating over the dataset...")
            memmap_iteration(memmap_data)
            print("Memmap iteration completed")