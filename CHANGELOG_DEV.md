# Changelog

| PR               | Type       | Ref. Issue(s) | Breaking Changes |PR Description|                                                                                  
|------------------|------------|---------------|------------------|------------------------------------------------------------------------------------------------|
| [#141](#pr-141-towards-stable-modalities-version)  | Bug Fix    |  [#129](https://github.com/Modalities/modalities/issues/129)         | **Yes**              | Towards stable modalities version                                                               |
| [#154](pr-154-manual-swiglu-implementation)  | Bug Fix    |  [#14](https://github.com/Modalities/modalities/issues/14)         | **Yes**              | Towards stable modalities version                                                               |
|    |   |           |        |                                                                |




## PR #141 Towards stable modalities version

This PR further stabilise the codebase and makes training more robust also w.r.t. loss spikes, which we fixed via scaled weight initialisation and an increased batch size in our experiments.
The PR also fixes all failing tests and adds a simple entrypoint for running cpu, single-gpu and multi-gpu tests. The PR contains multiple sub PRs. 

**General changes:**
* Bug fix: the model evaluation mode is now properly deactivated after evaluation (see PR [#131](https://github.com/Modalities/modalities/pull/131))
* Bug fix: Fixed the implementation of Pre-LN for GPT2 model (see PR [#136](https://github.com/Modalities/modalities/pull/136))
* Enhancement: Further mixed precision strategies; also added one matching MegatronLM's.
* Enhancement: Single, unified entrypoint for running cpu, single-gpu and multi-gpu tests. All tests fixed. (PR [#155](https://github.com/Modalities/modalities/pull/155))
* Enhancement: Previously, we would chunk the dataset into `block_size` long chunks. Each chunk would then be used for training individually. As a result, the last token of a block would be only used as a target but never as an input. We changed this, such that we reuse the last token of a batch as the first one of the subsequent batch. (PR [#158](https://github.com/Modalities/modalities/pull/158))
* Bug: Indexing of the original samples of the dataset pbin files had multiple bugs. The index tuples are now always in bytes and the start of the first sample in the data section starts at byte 0 (before the was a wrong offset) (PR [#164](https://github.com/Modalities/modalities/pull/164))
* Enhancement: Improvements on the current pull request template and addition of several issue templates (bug report, documentation, feature request, blank) (PR [#172](https://github.com/Modalities/modalities/pull/172))
* Components and factories for plain, scaled and scaled_embed initialisation. (PR [#161](https://github.com/Modalities/modalities/pull/161))
* in GPT2 model training configs, the standard deviation `std` can now be set to the string `auto` (in which case it will equal `sqrt(2/(5*hidden_dim))`, see e.g. https://arxiv.org/abs/2312.16903) (PR [#161](https://github.com/Modalities/modalities/pull/161))
* The CoCa model, which previously used a hardcoded, (probably not entirely correct) scaled initialization (see #165), can now only use plain initialization (PR [#161](https://github.com/Modalities/modalities/pull/161))


**Breaking changes:** 
* Enhancement: Logging is now always based on #training steps and #consumed tokens (PR [#137](https://github.com/Modalities/modalities/pull/137))
   This change is a breaking change and the experiment configs need to adapated as shown [here](https://github.com/Modalities/modalities/pull/137/files#diff-2bea5a6678ec91ea603cc2e80d17847360af5e9f7624c8e710f329ee1eb9b4f4). 
* Enhancement: The model parameters are now grouped within the respective model. The optimizer can leverage these groups to e.g., only apply weight decay to non-layer-norm weights. See [here](https://github.com/Modalities/modalities/pull/139/files#diff-2bea5a6678ec91ea603cc2e80d17847360af5e9f7624c8e710f329ee1eb9b4f4) for the necessary config changes. (PR [#139](https://github.com/Modalities/modalities/pull/139))
* Enhancement: We support now different attention implementations (manual, pytorch flash, DAO flash) See [here](https://github.com/Modalities/modalities/pull/138/files#diff-2bea5a6678ec91ea603cc2e80d17847360af5e9f7624c8e710f329ee1eb9b4f4) for the respective config changes. (PR [#138](https://github.com/Modalities/modalities/pull/138))
* Enhancement: replaced `block_size` in `Dataset`, `Model` and `NumberConversion` with `sequence_length` (PR [#158](https://github.com/Modalities/modalities/pull/158))
* Enhancement: `block_size` is now `sequence_length +1` and we should always specify `sequence_length` as a value of power of 2. (PR [#158](https://github.com/Modalities/modalities/pull/158))
* Enhancement: Restricted the codebase to the officially supported python versions 3.10 and 3.11 ((PR [#174](https://github.com/Modalities/modalities/pull/174)))
* All training configs require an additional component for initialization of the raw model (i.e. the model with random weights), as shown [here](https://github.com/Modalities/modalities/blob/7d26675051b918c3a2b98f32f50cb3ca8ef97d6f/config_files/training/config_lorem_ipsum.yaml#L181). (PR [#161](https://github.com/Modalities/modalities/pull/161))

## Checklist before submitting final PR
- [ ] My PR is minimal and addresses one issue / enhancement in isolation
- [ ] I have merged main into this feature branch
- [ ] I have reviewed my own code w.r.t. correct implementation, missing type hints, proper documentation, etc.
- [ ] I have run a sample config for model training
- [ ] I have fixed all failing tests (`python tests/tests.py`)



## PR #154 Manual SwiGLU implementation

This [PR](https://github.com/Modalities/modalities/pull/154) adds a manual SwiGLU implementation. The original one from xops was imcompatible with activation checkpointing (see issue [#14](https://github.com/Modalities/modalities/issues/14)) 

**General changes:**
* replaces xops swiglu imlementation with custom reimplementation

**Breaking changes:** 
* renaming of `fused_swiglu` to `swiglu` in `ActivationType` (see [here](https://github.com/Modalities/modalities/pull/154/commits/90fb3bd06a407333423cffeab486711e26ef8ddf) for the respective config changes)

## PR #236 Remove code related to Mamba

This [PR](https://github.com/Modalities/modalities/pull/236) removes all code related to Mamba. The latest state of main with Mamba can be found in the branch main_with_mamba.

**General changes:**
* Removes Mamba-related code

**Breaking changes:** 
* None
 

## PR #254 Warmstart infrastructure switch

This PR mainly addresses the warmstart of model training, e.g., after GPU crashes.

**General Changes**
* Fixes issue #242 
* Warmstarts with changing infrastructure (e.g.,. different number of GPUs) are now supported.
* Restructures the settings part of the configs to 
* Adds various checks for consistency of model training (e.g., target tokens and number of dataset tokens mismatch)
* Refactors all configs to be runnable again
* Adds an interactive jupyter notebook-based Tutorial on how to use Modalities. (merged from PR #239 )
* Adds a warmstart tutorial
* TrainingReportGenerator that creates a report on the training setup and prints out warnings in case of inconsistencies.
* Activation Checkpointing is now a component
* Added further NumberConversion routines

**Breaking Changes**
* the settings part of the configs have been completely refactored



## PR #261 Dataloader inefficiencies fix and combined dataset feature

This PR addresses issue #258 (inefficiencies in the dataloader) and additionally introduces a combined dataset, where a dataset can now comprise a list of datasets and iterate over them.
As part of fixing the dataloader inefficiencies, we now implement the sample skipping functionality not on the dataloader level  anymore but in an adapted version of the PyTorch `DistributedSampler`. I reran a warm start and the learning is equivalent to a full, non-warmstarted run. 

<img width="1415" alt="Screenshot 2024-09-27 at 10 36 19" src="https://github.com/user-attachments/assets/65dfb1ed-e96b-4f50-a127-bc9d240ddff9">


**General Changes**
* Introduced `ResumableDistributedSampler` which is a copy of the PyTorch `DistributedSampler` added with the feature to skip samples. This is from now on used for warmstarts instead of the `skip_num_samples` in the Dataloader. In case of skipping samples, the dataloader had to instantiate a `ResumableBatchSampler` which was internally iterating over all the dataset indices. For small datasets this was fine, but for larger datasets (in the trillion token range) this became a bottleneck at instantiation time:
https://github.com/Modalities/modalities/blob/b79d04d3e92d0845c5ec91f8dd41176fd543cb23/src/modalities/dataloader/samplers.py#L25-L28
Skipping in the  `ResumableDistributedSampler` is skipping in O(1) now. The `ResumableBatchSampler` was removed from the codebase.
* Replaced the packed index generation routine (inefficient due to for loop)
https://github.com/Modalities/modalities/blob/b79d04d3e92d0845c5ec91f8dd41176fd543cb23/src/modalities/dataloader/dataset.py#L331-L334
with a vectorized version.
* added new `NumberConversion` routine `num_samples_from_num_tokens `

**Breaking Changes**
* Removed RepeatingDataloader, as a feature that was never actively used for running multiple epochs and had complex maintenance when refactoring the sampling. If needed we could reimpliment it. 
*  In the settings, the `training_progress` section has now `num_seen_samples` instead of `local_num_seen_batches `, as skipping is now done on the Sampler level and not on the dataloader level anymore
* `batch_size ` and `fast_forward_batch_id ` fields in the `LLMDataLoader ` are not neede anymore and were removed.


## PR #269 Large file reader efficiency improvements and byte reading support

This PR makes the LargeFileLinesReader about 50% faster by using mmap instead of file seek operations. 
We can also now configure the encoding used for reading the documents. If encoding is specifically set to None (default is utf-8), we return the document as a byte string not enforcing any encoding. This is especially helpful when we e.g., sample from the data and want to create a subset of the dataset. In this case, we can just pass around the bytes representation. 


**Breaking Changes**
* None


## PR #280 Bug fix: the number of bytes per token were wrongly calculated

This PR fixes the bytes per token calculation.
Generally, we estimate how many bytes are needed to encode the full range of the vocabulary. 
E.g., for a vocab size > 65536, we need 3 bytes for each token in the pbin file. 

The calculation was wrong but coincidentally correct for the GPT2 tokenizer. 



## PR #281: Bug fix: The char-based index is not always consistent with the byte-based index.

The first character of the string "ø This is..." is written on disc as two bytes, namely \xc3\xb8, when encoded as utf-8. 
Therefore, the byte-based index has one more byte/char than the char-based index. 

For consistency, we don't consider any char-based indexes anymore and always refer to byte-based indexes. 


## PR #282: Bug fix: Enforce power of 2 number of bytes per token


Previously, the number of bytes per token was calculated by `math.ceil(log_2(vocab_size)/8)`, leading to ranges between 1 and 4 bytes. 
However, the dataset implementation only support 1, 2 and 4 bytes per token, as defined here

https://github.com/Modalities/modalities/blob/0483362abac93e45850e56adaea7921e96836d59/src/modalities/dataloader/dataset.py#L202-L206

and 

https://github.com/Modalities/modalities/blob/0483362abac93e45850e56adaea7921e96836d59/src/modalities/dataloader/dataset.py#L233-L234

I added a switch case that maps to the respective byte sizes, when packing the data.

This adds some inefficiencies as a vobabulary size > 65536 already requires 4 bytes per token, effectively doubling the storage requirements. 


## PR #283: Bug fix: Only append eod token once when packing / tokenizing

Some HF tokenisers such as `xlm-roberta-large` add special tokens (e.g., eod token) automatically when encoding text, whereas others, such as `gpt2`, do not add special tokens. 

This side-effect in the transformers library has lead to the eod token being appended twice when tokenizing / packing our data. We added a check for this and only append the eod token once now:
https://github.com/Modalities/modalities/blob/1c1ccdc973283c45bc8c9fadf4d20f03e435cd04/src/modalities/dataloader/create_packed_data.py#L327-L330

Additionally, I added a script that verifies the consistency of the indexation and tokenization of a given JSONL file. We run the indexation and tokenization routines in modalities and compare it to tokenized JSONL file to which we applied the HF tokenizer directly. 

## PR #379 Instruction Tuning Support

  * New entry point `apply_chat_template` to form chats and create index and pbin files of it
  * A wrapper for collate functions to include tokens in the loss which appear between indicator tokens
  * A new parameter for the PackedMemMapDatasetContinuous to allow not to re-use the last target token
  * A tutorial how to apply instruction-tuning on a Huggingface Model


## PR #359 Activation Checkpoint with FSDP2 

This PR adds activation checkpointing (AC) support for FSDP2. 
There are now three AC variants: 
* Full AC (same as before, where entire complete modules get ACed, leading to the largest memory footprint reduction)
* Selective Layer AC (only very nth layer or module is ACed)
* Selective OP Ac (only certain OPs, typically low memory but compute intense, are checkpointed)

## PR #374 Tensor Parallelism Support

* adds support for Tensor Parallelism (including Sequence Parallelism). 
* adds a debugging toolkit to track the input and output tensors during a forward pass, gradients during the backward pass and weight tensors.
Tensors can be either normal Tensors or DTensors.  
