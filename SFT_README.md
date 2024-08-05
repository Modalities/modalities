# Supervised Fine-tuning with Modalities

Currently supported are Instruction-tuning and Low-rank Adaption (LorA), as explained in more detail next.

## Instruction-tuning
* entry point to prepare data
* jinja2 templates
* The b_include_to_loss_token, e_include_to_loss_token are required to be part of each chat template for proper loss masking!
* hash to connect files
 
* truncation, padding
* re-use last target

### Create Prompts from Conversations
To prepare the instruction-tuning data we created a new entry point `apply_chat_template`, which requires a [configuration file](./config_files/data_preparation/apply_chat_template_config.yaml). Wihtin it we define:
* the path to instruction-tuning dataset as a JSONL file wereas each line contains a structured conversation as an array of dictionaries.
* A [jinja2](https://jinja.palletsprojects.com/en/3.1.x/) chat template which defines the rules how to glue `chat_template_data` and the data within the JSONL together to one `chat` string.

As part of the `chat_template_data`, we require the special tokens `b_include_to_loss_token` and `e_include_to_loss_token`. 
> ❗ You should choose sequences which are tokenized into a single token and will not appear in the assistant utterances of the instruction-tuning data!

They are used to mark the begin and end of the assistant turns, as we need to include only tokens between those into the loss computation during instruction-tuning with modalities.

```yaml
chat_template_data:
  ...
  special_tokens:
      b_include_to_loss_token: ^
      e_include_to_loss_token: $
```

Run the `apply_chat_template` entry point with:
```bash
modalities data apply_chat_template --config_file_path config_files/data_preparation/apply_chat_template_config.yaml
```

This will create two files
1. The new JSONL file with a new attribute `chat` containing the conversations e.g. `lorem_ipsum_sft_converted.aadd295.jsonl`
2. The config used to generate the `chat` e.g. `sft_chat_template_config.aadd295.yaml`

> Both files names contain the first 7 symbols of the hash of the config file, to group files which belong together!

### Create idx and pbin files
Before continuing with the instruction-tuning you need to index the created JSONL and convert it to a packed data file.

> Make sure to use the same hash for correct grouping when defining the output file names!

For example:
```bash
# create idx file
modalities data create_raw_index --index_path data/lorem_ipsum_sft_converted.aadd295.idx data/lorem_ipsum_sft_converted.aadd295.jsonl 

# create pbin file
modalities  data pack_encoded_data --config_file_path config_files/data_preparation/packed_chat_dataset_config.yaml
```

> The [packed_chat_dataset_config.yaml](config_files/data_preparation/packed_chat_dataset_config.yaml) must use truncation and padding!

### Instruction-Tuning

With your prepared instruction-tuning data as pbin file, you can now instruction-tune.

Make sure to use the wrapped collate function.

* You need to look up the `b_include_to_loss_token` and `e_include_to_loss_token` as defined within your `sft_chat_template_config.aadd295.yaml`. If configured the pbin creation correctly, you only need to check for matching hash suffixes.
* Set the `loss_ignore_index` which gets ignored by your loss function. In torch this is usually -100.
* We need a tokenizer to tokenize the `b_include_to_loss_token` and `e_include_to_loss_token`

For example (Copied from [config_files/training/config_lorem_ipsum_sft.yaml](config_files/training/config_lorem_ipsum_sft.yaml)):
```yaml
collate_fn:  
  component_key: collate_fn
  variant_key: mask_loss_collator_wrapper
  config:
    wrapped_collate_fn:  
      component_key: collate_fn
      variant_key: gpt_2_llm_collator
      config:
        sample_key: ${settings.referencing_keys.sample_key}
        target_key: ${settings.referencing_keys.target_key}
    target_keys_to_mask:
      - ${settings.referencing_keys.target_key}
    loss_ignore_index: -100
    mask_tokens:
      b_include_to_loss_token: ^
      e_include_to_loss_token: $
    tokenizer:
      instance_key: tokenizer
      pass_type: BY_REFERENCE
```

Finally, run the instruction-tuning with the `run` entry point:
```bash
torch.distributed.run --nnodes 1 --nproc_per_node 2 --rdzv-endpoint=0.0.0.0:29555 src/modalities/__main__.py run --config_file_path config_files/training/config_lorem_ipsum_sft.yaml
```

## Low-rank Adaption (LorA)

TBD
