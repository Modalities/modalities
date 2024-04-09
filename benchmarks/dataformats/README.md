
# Data Formats Benchmarking

This guide covers how to set up and run storage checks and benchmarks on different data formats. Our scripts are designed to evaluate storage efficiency and performance characteristics of various data formats using Python.

## Dependencies

- Python 3.x
- Scalene: A high-resolution memory and time profiler. For installation and more details, check out [Scalene on GitHub](https://github.com/plasma-umass/scalene).

## Running Storage Checks

To perform storage checks, execute the `storage_check.py` script located within the `dataformats` path. This script checks the storage conditions of the specified data paths.

```bash
python3 storage_check.py --webdata [WEBDATA_PATH] --memmap [MEMMAP_DATA_PATH]
```


## Running Data Benchmarking

```bash
scalene --- init_iter.py --webdata [WEBDATA_PATH] --memmap [MEMMAP_DATA_PATH]
```

## Results

The following results are carried out with the validation split.

### COCO dataset

| DataFormat | Preparation | Storage MB | Storage #Files | Init Time | Init Mem | Iter Time | Iter mem |
| ---------- | ----------- | ---------- | -------------- | --------- | -------- | --------- | -------- |
| WebDataset img+text| ?           | 793.54     | 5              | 2.01s     | 2.01     | 17.88     | 3.48     |
| Memmap   img+text  | ?           | 2305.35    | 1              | 2.78      | 64.33    | 35.89     | 0.02     |
