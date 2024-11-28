# Components

## Models

|Component type | Component Version  | Implementation | Configuration | Component Interface | Description |
|---------------|--------------------|----------------|---------------|---------------------|-------------|
| model | gpt2 | [GPT2LLM](../../src/modalities/models/gpt2/gpt2_model.py)| [GPT2LLMConfig](../../src/modalities/models/gpt2/gpt2_model.py) | [NNModel](../../src/modalities/models/model.py) | GPT2 model for language modeling |
| model | huggingface_pretrained_model | [HuggingFacePretrainedModel](../../src/modalities/models/huggingface/huggingface_model.py)| [HuggingFacePretrainedModelConfig](../../src/modalities/models/huggingface/huggingface_model.py) | [NNModel](../../src/modalities/models/model.py) | HuggingFace pretrained model for language modeling |
| model | checkpointed | [ModelFactory.get_checkpointed_model](../../src/modalities/models/model_factory.py)| [CheckpointedModelConfig](../../src/modalities/config/config.py) | [nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) | Checkpointed Model instance |
| model | fsdp_wrapped | [ModelFactory.get_fsdp_wrapped_model](../../src/modalities/models/model_factory.py)| [FSDPWrappedModelConfig](../../src/modalities/config/config.py) | [NNModel](../../src/modalities/models/model.py) | Model that has been sharded via FSDP |
| model | model_initialized | [ModelFactory.get_weight_initalized_model](../../src/modalities/models/model_factory.py)| [WeightInitializedModelConfig](../../src/modalities/config/config.py) | [nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) | Model with initialized weights |
| model | coca | [CoCa](../../src/modalities/models/coca/coca_model.py)| [CoCaConfig](../../src/modalities/models/coca/coca_model.py) | [NNModel](../../src/modalities/models/model.py) |[CoCa Model (Contrastive Captioners) ](https://arxiv.org/abs/2205.01917) |

## Weight Initialization

|Component type | Component Version  | Implementation | Configuration | Component Interface | Description |
|---------------|--------------------|----------------|---------------|---------------------|-------------|
| model_initialization | composed | [ComposedInitializationRoutines.get_composed_model_initializer](../../src/modalities/nn/model_initialization/composed_initialization.py)| [ComposedModelInitializationConfig](../../src/modalities/nn/model_initialization/composed_initialization.py) | [ModelInitializationIF](../../src/modalities/nn/model_initialization/initialization_if.py) | Component for initializing model weights in place |

## Losses

|Component type | Component Version  | Implementation | Configuration | Component Interface | Description |
|---------------|--------------------|----------------|---------------|---------------------|-------------|
| loss | clm_cross_entropy_loss | [CLMCrossEntropyLoss](../../src/modalities/loss_functions.py)| [CLMCrossEntropyLossConfig](../../src/modalities/config/config.py) | [Loss](../../src/modalities/loss_functions.py) | Cross-entropy loss function |

## Optimizers

|Component type | Component Version  | Implementation | Configuration | Component Interface | Description |
|---------------|--------------------|----------------|---------------|---------------------|-------------|
| optimizer | adam | [OptimizerFactory.get_adam](../../src/modalities/optimizers/optimizer_factory.py)| [AdamOptimizerConfig](../../src/modalities/config/config.py) | [Optimizer](../../src/modalities/models/model.py) | ADAM optimizer |
| optimizer | adam_w | [OptimizerFactory.get_adam_w](../../src/modalities/optimizers/optimizer_factory.py)| [AdamWOptimizerConfig](../../src/modalities/config/config.py) | [Optimizer](../../src/modalities/models/model.py) | ADAMW Optimizer |
| optimizer | checkpointed | [OptimizerFactory.get_checkpointed_optimizer](../../src/modalities/optimizers/optimizer_factory.py)| [CheckpointedOptimizerConfig](../../src/modalities/config/config.py) | [Optimizer](../../src/modalities/models/model.py) | Optimizer instantiated from checkpoint |

## LR Scheduling

|Component type | Component Version  | Implementation | Configuration | Component Interface | Description |
|---------------|--------------------|----------------|---------------|---------------------|-------------|
| scheduler | dummy_lr | [DummyLRScheduler](../../src/modalities/optimizers/lr_schedulers.py)| [DummyLRSchedulerConfig](../../src/modalities/config/config.py) | [LRScheduler](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate) | Fake lr scheduler not adapting the lr rate |
| scheduler | step_lr | [StepLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html)| [StepLRSchedulerConfig](../../src/modalities/config/config.py) | [LRScheduler](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate) | Decays the learning rate of each parameter group by gamma every step_size steps |
| scheduler | constant_lr | [ConstantLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ConstantLR.html#torch.optim.lr_scheduler.ConstantLR)| [ConstantLRSchedulerConfig](../../src/modalities/config/config.py) | [LRScheduler](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate) | Multiplies the learning rate of each parameter group by a small constant factor until the number of steps reaches a pre-defined milestone |
| scheduler | onecycle_lr | [OneCycleLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html#torch.optim.lr_scheduler.OneCycleLR)| [OneCycleLRSchedulerConfig](../../src/modalities/config/config.py) | [LRScheduler](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate) | Sets the learning rate of each parameter group according to the 1cycle learning rate policy. |
| scheduler | cosine_annealing_lr | [CosineAnnealingLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html#torch.optim.lr_scheduler.CosineAnnealingLR)| [CosineAnnealingLRSchedulerConfig](../../src/modalities/config/config.py) | [LRScheduler](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate) | Set the learning rate of each parameter group using a cosine annealing schedule |


## Tokenization

|Component type | Component Version  | Implementation | Configuration | Component Interface | Description |
|---------------|--------------------|----------------|---------------|---------------------|-------------|
| tokenizer | pretrained_hf_tokenizer | [PreTrainedHFTokenizer](../../src/modalities/tokenization/tokenizer_wrapper.py) | [PreTrainedHFTokenizerConfig](../../src/modalities/config/config.py) | [TokenizerWrapper](../../src/modalities/tokenization/tokenizer_wrapper.py) | Pretrained Huggingface tokenizer |
| tokenizer | pretrained_sp_tokenizer | [PreTrainedSPTokenizer](../../src/modalities/tokenization/tokenizer_wrapper.py) | [PreTrainedSPTokenizerConfig](../../src/modalities/config/config.py) | [TokenizerWrapper](../../src/modalities/tokenization/tokenizer_wrapper.py) | Pretrained SentencePiece tokenizer |

## Datasets

|Component type | Component Version  | Implementation | Configuration | Component Interface | Description |
|---------------|--------------------|----------------|---------------|---------------------|-------------|
| dataset | mem_map_dataset | [DatasetFactory.get_mem_map_dataset](../../src/modalities/dataloader/dataset_factory.py)| [MemMapDatasetConfig](../../src/modalities/config/config.py) | [Dataset](../../src/modalities/dataloader/dataset.py) | MemMap Dataset |
| dataset | packed_mem_map_dataset_continuous | [DatasetFactory.get_packed_mem_map_dataset_continuous](../../src/modalities/dataloader/dataset_factory.py)| [PackedMemMapDatasetContinuousConfig](../../src/modalities/config/config.py) | [Dataset](../../src/modalities/dataloader/dataset.py) | Packed Memory Mapped Dataset Continuous |
| dataset | dummy_dataset | [DatasetFactory.get_dummy_dataset](../../src/modalities/dataloader/dataset_factory.py)| [DummyDatasetConfig](../../src/modalities/dataloader/dataset.py) | [Dataset](../../src/modalities/dataloader/dataset.py) | Dummy dataset creating random samples of specified shape |
| dataset | combined | [DatasetFactory.get_combined_dataset](../../src/modalities/dataloader/dataset_factory.py)| [CombinedDatasetConfig](../../src/modalities/dataloader/dataset.py) | [Dataset](../../src/modalities/dataloader/dataset.py) | Dataset implementation combining multiple datasets into one. |

## Data sampling

|Component type | Component Version  | Implementation | Configuration | Component Interface | Description |
|---------------|--------------------|----------------|---------------|---------------------|-------------|
| sampler | distributed_sampler | [DistributedSampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler)| [DistributedSamplerConfig](../../src/modalities/config/config.py) | [Sampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler) | Sampler that restricts data loading to a subset of the dataset for distributed training |
| batch_sampler | default | [BatchSampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.BatchSampler) | [BatchSamplerConfig](../../src/modalities/config/config.py) | [Sampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler) | Wraps another sampler to yield a mini-batch of indices. |

## Data collation

|Component type | Component Version  | Implementation | Configuration | Component Interface | Description |
|---------------|--------------------|----------------|---------------|---------------------|-------------|
| collate_fn | gpt_2_llm_collator | [GPT2LLMCollateFn](../../src/modalities/models/gpt2/collator.py)| [GPT2LLMCollateFnConfig](../../src/modalities/config/config.py) | [CollateFnIF](../../src/modalities/models/gpt2/collator.py) | Data collator for the GPT2 model |
| collate_fn | coca_collator | [CoCaCollatorFn](../../src/modalities/models/gpt2/collator.py)| [CoCaCollateFnConfig](../../src/modalities/config/config.py) | [CollateFnIF](../../src/modalities/models/gpt2/collator.py) | Data collator for the CoCa model |

## Data loaders

|Component type | Component Version  | Implementation | Configuration | Component Interface | Description |
|---------------|--------------------|----------------|---------------|---------------------|-------------|
| data_loader | default | [DataloaderFactory.get_dataloader](../../src/modalities/dataloader/dataloader_factory.py)| [LLMDataLoaderConfig](s../../src/modalities/config/config.py) | [DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) | LLM Data loader extending pytorch data loader functionality |

## Checkpointing

|Component type | Component Version  | Implementation | Configuration | Component Interface | Description |
|---------------|--------------------|----------------|---------------|---------------------|-------------|
| checkpoint_saving | default | [CheckpointSaving](../../src/modalities/checkpointing/checkpoint_saving.py)| [CheckpointSavingConfig](s../../src/modalities/config/config.py) | -- | Component for saving checkpoints based on a savig and execution strategy. |
| checkpoint_saving_strategy | save_every_k_steps_checkpointing_strategy | [SaveEveryKStepsCheckpointingStrategy](../../src/modalities/checkpointing/checkpoint_saving_strategies.py)| [SaveEveryKStepsCheckpointingStrategyConfig](../../src/modalities/config/config.py) | [CheckpointSavingStrategyIF](../../src/modalities/checkpointing/checkpoint_saving_strategies.py) | Checkpointing strategy saving a checkpoint every k steps |
| checkpoint_saving_strategy | save_k_most_recent_checkpoints_strategy | [SaveKMostRecentCheckpointsStrategy](../../src/modalities/checkpointing/checkpoint_saving_strategies.py)| [SaveKMostRecentCheckpointsStrategyConfig](../../src/modalities/config/config.py) | [CheckpointSavingStrategyIF](../../src/modalities/checkpointing/checkpoint_saving_strategies.py) | Checkpointing strategy saving only the last k checkpoints and deleting the previous ones |
| checkpoint_saving_execution | fsdp | [FSDPCheckpointSaving](../../src/modalities/checkpointing/fsdp/fsdp_checkpoint_saving.py)| [FSDPCheckpointSavingConfig](../../src/modalities/config/config.py) | [CheckpointSavingExecutionABC](../../src/modalities/checkpointing/checkpoint_saving_execution.py) | FSDPCheckpointSaving class for saving checkpoints of FSDP models and optimizers. |
| checkpoint_loading | fsdp | [FSDPCheckpointLoading](../../src/modalities/checkpointing/fsdp/fsdp_checkpoint_loading.py)| [FSDPCheckpointLoadingConfig](../../src/modalities/config/config.py) | [CheckpointLoadingIF](../../src/modalities/checkpointing/checkpoint_loading.py) | Component for loading FSDP checkpoints|
| checkpoint_loading | torch | [TorchCheckpointLoading](../../src/modalities/checkpointing/torch/torch_checkpoint_loading.py)| [TorchCheckpointLoadingConfig](../../src/modalities/config/config.py) | [CheckpointLoadingIF](../../src/modalities/checkpointing/checkpoint_loading.py) | Component for loading PyTorch checkpoints|

## Logging

|Component type | Component Version  | Implementation | Configuration | Component Interface | Description |
|---------------|--------------------|----------------|---------------|---------------------|-------------|
| progress_subscriber | dummy | [ProgressSubscriberFactory.get_dummy_progress_subscriber](../../src/modalities/logging_broker/subscriber_impl/subscriber_factory.py)| [DummyProgressSubscriberConfig](../../src/modalities/config/config.py) | [MessageSubscriberIF](../../src/modalities/logging_broker/subscriber.py) | Dummy Progress subscriber not consuming any messages|
| progress_subscriber | rich | [ProgressSubscriberFactory.get_rich_progress_subscriber](../../src/modalities/logging_broker/subscriber_impl/subscriber_factory.py)| [RichProgressSubscriberConfig](../../src/modalities/config/config.py) | [MessageSubscriberIF](../../src/modalities/logging_broker/subscriber.py) | Subscriber for writing out rich-formatted console outputs w.r.t. to training and evaluation progress |
| results_subscriber | wandb | [ProgressSubscriberFactory.get_wandb_result_subscriber](../../src/modalities/logging_broker/subscriber_impl/subscriber_factory.py)| [WandBEvaluationResultSubscriberConfig](../../src/modalities/config/config.py) | [MessageSubscriberIF](../../src/modalities/logging_broker/subscriber.py) | Subscriber for logging evaluation results to Weights and Biases |

## Layer Norms

|Component type | Component Version  | Implementation | Configuration | Component Interface | Description |
|---------------|--------------------|----------------|---------------|---------------------|-------------|
| layer_norm | rms_norm | [RMSLayerNorm](../../src/modalities/models/components/layer_norms.py)| [RMSLayerNormConfig](../../src/modalities/models/components/layer_norms.py) | [nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) | RMS Layer norm |
| layer_norm | layer_norm | [nn.LayerNorm](../../src/modalities/models/components/layer_norms.py)| [LayerNormConfig](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) | [nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) | Layer norm |

## Gradient Clipping

|Component type | Component Version  | Implementation | Configuration | Component Interface | Description |
|---------------|--------------------|----------------|---------------|---------------------|-------------|
| gradient_clipper | fsdp | [FSDPGradientClipper](../../src/modalities/training/gradient_clipping/fsdp_gradient_clipper.py)| [FSDPGradientClipperConfig](../../src/modalities/training/gradient_clipping/fsdp_gradient_clipper_config.py) | [GradientClipperIF](../../src/modalities/training/gradient_clipping/gradient_clipper.py) | FSDP Gradient Clipper |
| gradient_clipper | fsdp_logging_only | [FSDPLoggingOnlyGradientClipper](../../src/modalities/training/gradient_clipping/fsdp_gradient_clipper.py)| [FSDPGradientClipperConfig](../../src/modalities/training/gradient_clipping/fsdp_gradient_clipper_config.py) | [GradientClipperIF](../../src/modalities/training/gradient_clipping/gradient_clipper.py) | Clipper that is responsible for logging the gradient norms without actually clipping the gradients |
| gradient_clipper | dummy | [DummyGradientClipper](../../src/modalities/training/gradient_clipping/fsdp_gradient_clipper.py)| [DummyGradientClipperConfig](../../src/modalities/training/gradient_clipping/fsdp_gradient_clipper_config.py) | [GradientClipperIF](../../src/modalities/training/gradient_clipping/gradient_clipper.py) | Dummy clipper that does not apply any gradient clipping. |

## Number conversions

|Component type | Component Version  | Implementation | Configuration | Component Interface | Description |
|---------------|--------------------|----------------|---------------|---------------------|-------------|
| number_conversion | local_num_batches_from_num_samples | [NumberConversion.get_local_num_batches_from_num_samples](../../src/modalities/utils/number_conversion.py)| [LocalNumBatchesFromNumSamplesConfig](../../src/modalities/utils/number_conversion.py) | -- | Calculates the number of local batches for each rank, given the global number of samples and number of ranks. |
| number_conversion | local_num_batches_from_num_tokens | [NumberConversion.get_local_num_batches_from_num_tokens](../../src/modalities/utils/number_conversion.py)| [LocalNumBatchesFromNumTokensConfig](../../src/modalities/utils/number_conversion.py) | -- | Calculates the number of local batches for each rank, given the global number of tokens and number of ranks. |
| number_conversion | local_num_batches_from_num_tokens | [NumberConversion.get_num_samples_from_num_tokens](../../src/modalities/utils/number_conversion.py)| [NumSamplesFromNumTokensConfig](../../src/modalities/utils/number_conversion.py) | -- | Calculates the number of global samples, given the global number of tokens and sequence length |
| number_conversion | num_steps_from_num_samples | [NumberConversion.get_num_steps_from_num_samples](../../src/modalities/utils/number_conversion.py)| [NumStepsFromNumSamplesConfig](../../src/modalities/utils/number_conversion.py) | -- | Calculates the number of steps given the global number of samples, local micro batch size and number of ranks. |
| number_conversion | num_steps_from_num_tokens | [NumberConversion.get_num_steps_from_num_tokens](../../src/modalities/utils/number_conversion.py)| [NumStepsFromNumTokensConfig](../../src/modalities/utils/number_conversion.py) | -- | Calculates the number of steps given the global number of tokens, local micro batch size and number of ranks. |
| number_conversion | num_tokens_from_num_steps | [NumberConversion.get_num_tokens_from_num_steps](../../src/modalities/utils/number_conversion.py)| [NumTokensFromNumStepsConfig](../../src/modalities/utils/number_conversion.py) | -- | Calculates the number of tokens from the number of steps, number of ranks, local micro batch size, global number of tokens, squence length and gradient accumulation steps |
| number_conversion | last_step_from_checkpoint_path | [NumberConversion.get_num_seen_steps_from_checkpoint_path](../../src/modalities/utils/number_conversion.py)| [NumberConversionFromCheckpointPathConfig](../../src/modalities/utils/number_conversion.py) | -- | Get the last step id from a model or checkpoint file path. |
| number_conversion | global_num_target_tokens_from_checkpoint_path | [NumberConversion.get_global_num_target_tokens_from_checkpoint_path](../../src/modalities/utils/number_conversion.py)| [NumberConversionFromCheckpointPathConfig](../../src/modalities/utils/number_conversion.py) | -- | Get the number of target tokens from a model or checkpoint file path. |
| number_conversion | num_tokens_from_packed_mem_map_dataset_continuous | [NumberConversion.get_num_tokens_from_packed_mem_map_dataset_continuous](../../src/modalities/utils/number_conversion.py)| [NumTokensFromPackedMemMapDatasetContinuousConfig](../../src/modalities/utils/number_conversion.py) | -- | Get the number of tokens stored in a [packed mem map continuous dataset](../../src/modalities/dataloader/dataset.py) from the respective dataset file path. |
| number_conversion | num_steps_from_raw_dataset_index | [NumberConversion.get_num_steps_from_raw_dataset_index](../../src/modalities/utils/number_conversion.py)| [NumStepsFromRawDatasetIndexConfig](../../src/modalities/utils/number_conversion.py) | -- | Get the number of steps partially from the raw index of a raw JSONL dataset. Requires the file path to index, number of ranks, local micro batch size and gardient accumulation steps. |