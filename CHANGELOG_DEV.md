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


## PR #389 Benchmark Tooling 
* adds benchmarking tooling to modalities and allows for scaling benchmarks across varying number of nodes and the cartesian product of configurable hyper parameters.

**Breaking Changes**
* Renaming: EvaluationResultToDiscSubscriberConfig.output_path -> EvaluationResultToDiscSubscriberConfig.output_file_path



## PR #410 MFU incorporates dp_degree now instead of world_size

This PR fixes the MFU and throughput calculations by taking the dp degree into account instead of the world size. When we use parallelization strategies on top of FSDP, then the world size is different from the  data parallel degree. This needs to be reflected in throughput and MFU metric calculations, as done by this PR. 

**Breaking Changes**
* Existing configs need to be adapted to correctly use dp degree rather than world size. 


## PR #425 Monitoring improvements
This PR improves training monitoring and logging across runs besides some other changes we did along while testing out scalability.

**General Changes**
* Configurable multi-layer FSDP units
* Option to provide experiment root path to modalities
* Added steppable profiler (e.g., for tracing of forward/backward passes)
* Fix: Hybrid sharding now correctly configurable
* Completely refactored the Profiling 
* Improved error handling. Errors are now captured and stored as JSON
* Add tutorials on Einsum Transformer (Example model integration) and profiling

**Breaking Changes**
* experiments_root_path is now exposed on an API level

## PR #XXX Quality-based document selection and up/downsampling

This PR adds a way to build a training blend by filtering documents on quality signals
and choosing how heavily each dataset is sampled, plus a fast way to see the token
budget a given selection yields before committing to a tokenization run.

**Motivation**

Two kinds of quality signal exist in practice: metrics a corpus already carries in its
own records (`fw_edu_scores`, `proxy_score`, `finemath_scores`, `perplexity`, ...), and
external per-document annotations that have to be joined on. Neither could be used to
shape a blend, and the only way to change a dataset's share was to duplicate it on disk
-- which is what the `_epoch_1` / `_epoch_2` directory convention did.

**General changes**

* New `modalities quality` command group with one subcommand per stage:
  `calibrate`, `build-sidecar`, `join-annotations`, `build-cube`, `preview`, `apply`
  and `write-packing-configs`.
* New `src/modalities/dataloader/preprocessing/quality/` package:
  * `registry` declares each dataset's source and how it joins to annotations. Four key
    kinds are supported, covering corpora that store a plain id, an id wrapped in
    `<urn:uuid:...>`, no id at all (keyed by a hash of the text), and a `<file>/<line>`
    pointer into a separate source corpus.
  * `tokens` measures per-dataset token estimators. Estimates are per document and
    based on the text rather than the stored line, because quality correlates with
    length and several corpora keep multiple renderings of a document in one record.
  * `sidecar` streams each JSONL once and records one row per document: position,
    length, estimated tokens, join key and native metrics.
  * `annotation_join` joins annotations by bucketing both sides on a hash of the key,
    so a split of billions of rows never needs a single hash table. Coverage,
    duplicate keys and unmatched documents are reported rather than hidden.
  * `cube` aggregates a sidecar into grouped document and token counts, which is what
    makes `preview` return in microseconds. Thresholds landing on a bin edge are exact;
    one landing inside a bin is reported as interpolated instead of silently guessed.
  * `selection` evaluates a YAML selection over both annotation labels and native
    metrics, with ordinal scales declared explicitly.
  * `materialize` writes the selection out as filtered `.idx` files.
* New `WeightedCombinedDataset` (component `dataset`/`weighted_combined`), which takes a
  float repeat factor per dataset. A ratio of 2.5 draws a dataset two and a half times
  per epoch and 0.3 draws three tenths of it, without duplicating anything on disk. The
  partial pass is chosen by a seeded affine permutation, so it is deterministic across
  ranks and restarts and spreads across the whole dataset rather than taking a prefix.
* New `TokenizerInstantiationModel`, so a tool can reuse a packing config for its
  tokenizer without also having to satisfy that config's `settings`.

**Notes**

Selection produces a filtered index rather than a filtered copy of the corpus:
`PackedDataGenerator` already tokenizes exactly the documents its index lists, so
`pack_encoded_data` consumes the output unchanged and no packing code was touched. An
ablation therefore costs megabytes of index rather than a second copy of the data, and
the source tree is never written to.

**Breaking Changes**

None. `CombinedDataset` and every existing config keep working as before.


## PR #XXX Quality selection: performance and sharding

Follow-up to the quality-selection PR, from profiling the pipeline against the real
43 TB blend rather than a fixture. Three of the four stages were far slower than they
needed to be, and two of them could not be parallelised at all.

**General changes**

* Native metrics declared as a plain field path (`.fw_edu_scores`,
  `.metadata.dclm_plus2."__label__1"`) are now read by direct dictionary lookup instead
  of jq. `jq.compile(...).input_value(record)` re-serialises the whole document on every
  call, which on a 21 KB record cost more than twenty times the rest of building a
  sidecar row; the full pass measured **29 MB/s with jq against 374 MB/s without, 13x
  end to end**. Patterns jq cannot reduce to a field chain still use jq, and
  `build-sidecar` now warns when one does, since that pattern then dominates the stage.
* `build_cube` groups with Arrow's C++ kernels instead of a Python loop over documents,
  batching row groups so cardinality saturates before each grouping pass:
  **246k -> 2.39M rows/s, 9.9x**. For the full blend that is ~50 minutes rather than
  ~8.5 hours, and the stage was not shardable, so it was a hard floor.
* `build-sidecar` takes `--shard_id`/`--num_shards`. Work is divided per JSONL file
  across every selected dataset, so one array covers the whole blend however unevenly
  the file counts fall. Previously the only split was per dataset, leaving a floor of
  the slowest dataset -- ~71 h for `finepdfs-en`.
* Annotation bucketing moved out of `join-annotations` into its own shardable
  `bucket-annotations` stage. Each task writes its own file per bucket and the join
  reads all of them, so the result is identical to a single-task run. Bucketing 13.9 bn
  annotation rows was ~32 h serial, with a ~8.7 h floor from the largest single split.
* `join-annotations` refuses to run against an incomplete bucketing run rather than
  silently dropping the annotations a missing task was carrying, which would have looked
  exactly like a corpus that was never annotated.

**Notes**

Measured on this cluster: sequential read from `/data` is ~282 MB/s per stream and
~3.8 GB/s aggregate. With the jq fix the sidecar pass is I/O-bound, so beyond ~16
concurrent tasks the storage is the limit rather than the code. End to end the one-time
setup goes from ~88 h to ~4 h on two nodes; previewing a selection is unaffected at
~10 s for the whole blend, because it only ever reads the cubes.

**Breaking Changes**

* `modalities quality build-sidecar` no longer takes `--file_id`; use
  `--shard_id`/`--num_shards`.
* `modalities quality join-annotations` no longer buckets. Run
  `modalities quality bucket-annotations` first. Its `--num_buckets` and
  `--rebuild_buckets` options moved to that command (`--rebuild_buckets` is now
  `--force`).
