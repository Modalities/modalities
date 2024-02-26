# Getting Started Tutorial

In this tutorial, we train a 60M-parameter GPT model on the [Redpajama V2](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2) dataset. 
As a preparation step, we already downloaded the Redpajama V2 sample dataset from Huggingface. Additionally, we sampled a subset of 512 documents for the training and evaluation split, respectively, and stored each of the splits as a jsonl file.

As a reference, this example has the following folder structure. Folders in <> will be populated as we go through this tutorial.  
```
└── getting_started
    ├── checkpoints
    │   └─ <checkpoint_folders>
    ├── example_config.yaml
    ├── data
    │   ├── mem_map
    │   │   └<preprocessed dataset files>
    │   └── raw
    │       ├── redpajama_v2_samples_512_test.jsonl
    │       └── redpajama_v2_samples_512_train.jsonl
    ├── getting_started_example.md
    ├── tokenizer
    │   └── tokenizer.json
    └── wandb
        └── <wandb_logs>
```

## 1. Preprocessing
A single line of the Redpajama V2 JSONL file has the structure denoted below. Since we are not interested in the meta data and quality signals for this minimal example, we consider the `raw_content` from each line without any filtering.
for model training. 
```json
{
   "raw_content":"Archivio Tag: 25 aprile\nSupermercati aperti 25 aprile 2019: centri commerciali e negozi a Roma, Milano, Napoli e Torino\nNell\u2019articolo odierno troverete tutte le informazioni utili su quali saranno i supermercati e le attivit\u00e0 commerciali che resteranno aperti in occasione...\nAuguri di Buon 25 Aprile 2017: frasi e pensieri originali sulla Festa della Liberazione",
   "doc_id":"2023-06\/0003\/it_head.json.gz\/1330",
   "meta":"{\"url\": \"http:\/\/www.correttainformazione.it\/tag\/25-aprile\", \"partition\": \"head_middle\", \"language\": \"it\"...}",
   "quality_signals":"{\"ccnet_length\": [[0, 1257, 1257.0]], \"ccnet_original_length\": [[0, 1257, 5792.0]], \"ccnet_nlines\": [[0, 1257, 11.0]], \"ccnet_origi..."
}
```
The two raw dataset splits for training and evaluation can be found in 
`data/raw/redpajama_v2_samples_512_train.jsonl` and `data/raw/redpajama_v2_samples_512_test.jsonl`
and need to be preprocessed into the [MemMap dataset format](https://github.com/Modalities/modalities/blob/main/src/modalities/dataloader/dataset.py). 
Firstly, we create the dataset index via

```sh
cd modalities/examples/getting_started/

# train split
modalities create_memmap_index --index_path data/mem_map/redpajama_v2_samples_512_train.idx \
                               data/raw/redpajama_v2_samples_512_train.jsonl

# test split
modalities create_memmap_index --index_path data/mem_map/redpajama_v2_samples_512_test.idx \
                               data/raw/redpajama_v2_samples_512_test.jsonl
```
In this step, we read the JSON file as a binary file, iterate over all characters und build up the sample index (char-wisestart and end position for each JSON sample)
as determined by the `\n` character positions. The sample index is stored in the specified `index_path`. Internally, the `create_memmap_index` command 
instantiates and calls the the [IndexGenerator](https://github.com/Modalities/modalities/blob/main/src/modalities/dataloader/create_index.py#L14).

After having determined the index, we create the packed dataset as described below by leveraging the tokenizer, jsonl file and the created index.

```sh
# train split
modalities create_packed_data --jq_pattern .raw_content \
                              --index_path data/mem_map/redpajama_v2_samples_512_train.idx \
                              --dst_path data/mem_map/redpajama_v2_samples_512_train.pbin \
                              --tokenizer_file tokenizer/tokenizer.json \
                              data/raw/redpajama_v2_samples_512_train.jsonl

# test split
modalities create_packed_data --jq_pattern .raw_content \
                              --index_path data/mem_map/redpajama_v2_samples_512_test.idx \
                              --dst_path data/mem_map/redpajama_v2_samples_512_test.pbin \
                              --tokenizer_file tokenizer/tokenizer.json \
                              data/raw/redpajama_v2_samples_512_test.jsonl
```
This will create the following file structure which can we can directly load into the [PackedMemMapdataset](https://github.com/Modalities/modalities/blob/main/src/modalities/dataloader/dataset.py#L65).
```
data/mem_map/
    redpajama_v2_samples_512_train.idx
    redpajama_v2_samples_512_train.pbin
    redpajama_v2_samples_512_test.idx
    redpajama_v2_samples_512_test.pbin
```

Technically, packed datasets are defined a self-contained format that stores the position of the sample

**Packed MemMap File Format**

```
|--8-BYTES-HEADER--|-------------------DATA-SEGMENT-------------------|----INDEX-SEGMENT----|


8 bytes header:
===============
specifies the size of the data segment in bytes. Since the header size is fixed to 8 bytes, 
the start and end position of each segment (i.e, header, data, index) is specified. Therefore, the theoretical maximum size of the data segment 
is 2^64 bytes = 18,446 peta bytes or 4600e+15 tokens or 4.6 quintillion tokens, given that a token has 4 bytes.


Data segment:
=============
contains the concatenated token ids for all documents.


Index segment:
==============
The index contains a tuple for each document with the format (byte_offset, segment_length),
where the byte_offset specifies the byte position in the data segment for the start of the document and segment_length. 
Therfore, the index segment would look like [(8, 100), (108, 302), (410, 803), ...]. The first sample starts at byte position 8 and
has a length of 100 bytes. The second sample therefore starts at byte position 108 and has  a length of 284 bytes and so on.
```

We have implemented different packing strategies on top of the file format, each making sure that a batch is completely filled up with documents without any trailing padding in the sequences.
In this tutorial, we use the simplest one with [PackedMemMapDatasetContinuous](https://github.com/Modalities/modalities/blob/main/src/modalities/dataloader/dataset.py#L115), which concatenates all documents as a sequence stream
first and then divides it into chunks of size context-length.   



In modalities, we describe the entire training and evaluation setup (i.e., components such das model, trainer, evaluator, dataloder etc.) within a single config file. Not only does this increase reproducibility but also allows for having the entire training runs under version control. 

The example config file for this experiment can be found in `examples/mem_map_redpajama_gpt/config_example_mem_map_dataset.yaml`. 

## 2. Training

Having created the dataset and defined the experiment in the configuration file, we can already start the training by running the following command.

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --rdzv-endpoint localhost:29505 \
                                              --nnodes 1 \
                                              --nproc_per_node 8 \
                                              $(which modalities) run --config_file_path example_config.yaml
```

The command can be broken down into the following parts:

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

## Evaluation

Given a checkpoint and tokenizer, we can load the model for text generation as follows

```sh
modalities generate_text --tokenizer_file tokenizer/tokenizer.json \
                        checkpoints/2024-01-15__14-02-37/eid_2024-01-15__14-02-37-model-num_samples_768.bin \
                         example_config.yaml 
```
which opens an interactive chatting CMD interface.

```
enter prompt> Once upon a time, 
there was ... <eos>
```