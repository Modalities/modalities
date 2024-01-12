# Getting Started Tutorial

In this tutorial, we train a 60M-parameter GPT model with [Redpajama V2](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2). 
As a preparation step, we already downloaded the Redpajama V2 sample dataset from Huggingface, We sampled a subset of 512 documents for the training and evaluation split, respectively, and stored each of the splits as a jsonl file.

A single line of the jsonl file has the structure denoted below. For training and  
```json
{
   "raw_content":"Archivio Tag: 25 aprile\nSupermercati aperti 25 aprile 2019: centri commerciali e negozi a Roma, Milano, Napoli e Torino\nNell\u2019articolo odierno troverete tutte le informazioni utili su quali saranno i supermercati e le attivit\u00e0 commerciali che resteranno aperti in occasione...\nAuguri di Buon 25 Aprile 2017: frasi e pensieri originali sulla Festa della Liberazione",
   "doc_id":"2023-06\/0003\/it_head.json.gz\/1330",
   "meta":"{\"url\": \"http:\/\/www.correttainformazione.it\/tag\/25-aprile\", \"partition\": \"head_middle\", \"language\": \"it\"...}",
   "quality_signals":"{\"ccnet_length\": [[0, 1257, 1257.0]], \"ccnet_original_length\": [[0, 1257, 5792.0]], \"ccnet_nlines\": [[0, 1257, 11.0]], \"ccnet_origi..."
}
```
The two raw datasets splits can be found in 
`data/raw/redpajama_v2_samples_512_train.jsonl` and `data/raw/redpajama_v2_samples_512_test.jsonl`

and need to be preprocessed into the MemMap datset format. 
Firstly, we create the dataset index via
```bash
cd modalities/examples/mem_map_redpajama_gpt

# train split
modalities create_memmap_index --index_path data/mem_map/redpajama_v2_samples_512_train.idx data/raw/redpajama_v2_samples_512_train.jsonl

# test split
modalities create_memmap_index --index_path data/mem_map/redpajama_v2_samples_512_test.idx data/raw/redpajama_v2_samples_512_test.jsonl
```
and then create the packed dataset as described below, leveraging the tokenizer, jsonl file and the created index.

```bash
# train split
modalities create_packed_data --jq_pattern .raw_content --index_path data/mem_map/redpajama_v2_samples_512_train.idx --dst_path data/mem_map/redpajama_v2_samples_512_train.pbin --tokenizer_file tokenizer/tokenizer.json data/raw/redpajama_v2_samples_512_train.jsonl

# test split
modalities create_packed_data --jq_pattern .raw_content --index_path data/mem_map/redpajama_v2_samples_512_test.idx --dst_path data/mem_map/redpajama_v2_samples_512_test.pbin --tokenizer_file tokenizer/tokenizer.json data/raw/redpajama_v2_samples_512_test.jsonl
```

This will create the following file structure which can we can directly load into the PackedMemMapdataset.
```
data/mem_map/
    redpajama_v2_samples_512_train.idx
    redpajama_v2_samples_512_train.pbin
    redpajama_v2_samples_512_test.idx
    redpajama_v2_samples_512_test.pbin
```

In modalities, we describe the entire training and evaluation setup (i.e., components such das model, trainer, evaluator, dataloder etc.) within a single config file. Not only does this increase reproducibility but also allows for having the entire training runs under version control. 

The example config file for this experiment can be found in `examples/mem_map_redpajama_gpt/config_example_mem_map_dataset.yaml`. 

Having created the dataset and defined the experiment in the configuration file, we can already start the training by running the following command.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --rdzv-endpoint localhost:29505 --nnodes 1 --nproc_per_node 8 $(which modalities) run --config_file_path config_example_mem_map_dataset.yaml
```

The command breaking

1. **`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`**:
   - Sets the `CUDA_VISIBLE_DEVICES` environment variable to a list of GPU device IDs (`0` to `7`). This restricts the visible CUDA devices to these specific GPUs.

2. **`torchrun`**:
   - A utility for running PyTorch distributed applications. It initializes the distributed environment and launches the application across multiple nodes and GPUs.

3. **`--rdzv-endpoint localhost:29505`**:
   - Specifies the rendezvous endpoint with `localhost:29505`. In distributed training, this process allows processes to exchange information at the start.

4. **`--nnodes 1`**:
   - Indicates the number of nodes (machines) to use for the training, set to `1` here, implying single-node distributed training.

5. **`--nproc_per_node 8`**:
   - Specifies the number of processes to launch on each node, with `8` processes corresponding to the 8 GPUs.

6. **`$(which modalities)`**:
   - Uses shell command substitution to find the path of the `modalities` executable. `which modalities` shows the full path of the `modalities` command.

7. **`run`**:
   - Command argument for the `modalities` executable to initiate the training.

8. **`--config_file_path config_example_mem_map_dataset.yaml`**:
   - Specifies the path to the configuration file. The file `config_example_mem_map_dataset.yaml` contains mentinoed configuratino of the components, including dataset and model configurations, training parameters, etc.


Already during the training, the checkpoints can be found locally in `checkpoints/` and the loss and metric developments can be inspected online in [Weights&Biases](https://wandb.ai/). 