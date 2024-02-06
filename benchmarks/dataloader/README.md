# Benchmarking of Dataset Implementations

## Motivation
We want to include a storage efficient, fast and generic dataset implementation in this repository.
Previous work and ideas were based on MegatronLM and its dataset implementation.

Unfortunately its usage is quite intransparent and causes regularly unexpected side effects.
Those problems are hard to trace, as we are not the original authors of the code.

Therefore we want to provide an own implementation, which comes with all the above mentioned benefits.
Most importantly, it should be at least as fast as MegatronLM's implementation.


## Benchmark Overview

We want to evaluate multiple aspects of the dataset implementations:
* preparation speed - All datasets need to do some initial steps like tokenization and indexing.
* initialization speed - When firing up a respective `Dataset` object inside the code.
* iteration speed - When accessing elements (in a random order) in the respective datasets


## Used Example Dataset

The experiments were conducted on a small sample of openwebtext. The data is provided in `.jsonl`-format.
The relevant data included can be found under `"text"` and is obviously text-only.
Each dataset with X samples refers to the first X lines in the full openwebtext data,
 as it can be obtained from huggingface.


## Experimental Setup

We relied on the functions provided in `launch_benchmark.sh`. One can reproduce those by calling e.g.

```shell
. launch_benchmark.sh

INPUT_DIR=<path-to-your-example-dataset.jsonl>

echo "MegatronLM:"
measure_megatronLM_iteration
echo "Modalities:"
measure_modalities_iteration
```

> For launching the preparation of MegatronLM's dataset, refer to:
> https://github.com/OpenGPTX/opengptx_data/tree/docs/modalities-vs-megatronlm-dl and look at the `launch_benchmark.sh`
> script.

#### Glossary

* **preparation:** refers here to the task of turning raw data (e.g. jsonl encoded text) into a binary file,
  which is loadable later for training. 
  For MegatronLM this means tokenizing and packing everything according to their defined format.
  For Modalities it means, indexing the raw data and packing it afterwards as token-ids.
* **initialization:** refers to the process of initializing a python object, 
  which represents the respective dataset (mostly represented via the `torch.Dataset`-interface)
* **iteration:** refers to process of iterating over the respective datasets - once sequentially and once shuffled.

## Results


| Evaluation Aspect    | Implementation |   Required Time    | # Samples in Data |
|----------------------|----------------|:------------------:|-------------------|
| preparation speed    | MegatronLM     | `0 min 16.965 sec` | `20000(OWT)`      |
| preparation speed    | Modalities     | `0 min 13.904 sec` | `20000(OWT)`      |
| preparation speed    | MegatronLM     | `2 min 11.856 sec` | `200000(OWT)`     |
| preparation speed    | Modalities     | `0 min 38.738 sec` | `200000(OWT)`     |
| initialization speed | MegatronLM     |    `19.3 msec`     | `20000(OWT)`      |
| initialization speed | Modalities     |    `5.85 msec`     | `20000(OWT)`      |
| initialization speed | MegatronLM     |    `180 msec `     | `200000(OWT)`     |
| initialization speed | Modalities     |     `58 msec`      | `200000(OWT)`     |
| iteration speed      | MegatronLM     |    `52.4 msec`     | `20000(OWT)`      |
| iteration speed      | Modalities     |    `66.8 msec`     | `20000(OWT)`      | 
| iteration speed      | MegatronLM     |    `426 msec `     | `200000(OWT)`     |
| iteration speed      | Modalities     |     `545 msec`     | `200000(OWT)`     |


