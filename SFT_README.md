# Supervised Fine-tuning with Modalities

Currently supported are Instruction-tuning and Low-rank Adaption (LorA), as explained in more detail next.

## Instruction-tuning
> Instruction-tuning currently only works with fast and slow Huggingface tokenizers, as the special tokens need to be added to the tokenizer.

The goal of instruction-tuning is to let the model learn instruction-following capabilites, so that it acts as an helpful assistant within an chat environment.
For this, we need to further fine-tune the model on conversational instruction data.
Specifically, we need the model to only learn to act as the assistant. Hence, we allow full attention on conversation, but calculate the loss only on the assistants untterances. 

For example, we only calculate the loss for the words not struck-trough:

> ~~You are Mody, a helpful assistant trained by the modalities team. Answer friendly and informatively to the user's messages.\nUser1: What is the best way to learn a new language?\nAssistant:^~~The best way to learn a new language is to practice regularly, immerse yourself in the language, and use a variety of resources like books, apps, and language classes. It's also helpful to practice with native speakers.\n°~~$User1: Thank you for the advice.\nAssistant:^~~You're welcome! Learning a new language can be a rewarding experience. If you have any more questions, feel free to ask.\n°~~$~~

### Create Prompts from Conversations
To prepare the instruction-tuning data we created a new entry point `apply_chat_template`, which requires a [configuration file](./config_files/data_preparation/apply_chat_template_config.yaml). Wihtin it we define:
* the path to instruction-tuning dataset as a JSONL file wereas each line contains a structured conversation as an array of dictionaries.
* A [jinja2](https://jinja.palletsprojects.com/en/3.1.x/) chat template which defines the rules how to glue `chat_template_data` and the data within the JSONL together to one `chat` string.

As part of the `chat_template_data`, we require the special tokens `b_include_to_loss_token` and `e_include_to_loss_token`. 
To prepare the instruction-tuning data we created a new entry point `apply_chat_template`, which requires a [configuration file](./config_files/data_preparation/apply_chat_template_config.yaml). Within it we define the path to instruction-tuning dataset as a JSONL file, in which each line contains a structured conversation as an array of dictionaries.

##### Example

Input JSONL file entry:
```json
{
    "id": 16,
    "conversations": [
        {
            "from": "human_1",
            "value": "What is the best way to learn a new language?"
        },
        {
            "from": "gpt",
            "value": "The best way to learn a new language is to practice regularly, immerse yourself in the language, and use a variety of resources like books, apps, and language classes. It's also helpful to practice with native speakers."
        },
        {
            "from": "human_1",
            "value": "Thank you for the advice."
        },
        {
            "from": "gpt",
            "value": "You're welcome! Learning a new language can be a rewarding experience. If you have any more questions, feel free to ask."
        }
    ]
}
```

Config:
```yaml
settings:
  src_path: data/lorem_ipsum_sft.jsonl
  dst_path: data/lorem_ipsum_sft_converted.jsonl
  conversations_key: conversations

instruction_data_transformation:
  role_mapping:
    human_1: User1
    human_2: User2
    gpt: Assistant

...

chat_template_data:
  ...
  special_tokens:
      b_include_to_loss_token: ^
      e_include_to_loss_token: $
```

Created JSONL file entry:
```json
{
    "id": 16,
    "conversations": [
        {
            "from": "User1",
            "value": "What is the best way to learn a new language?"
        },
        {
            "from": "Assistant",
            "value": "The best way to learn a new language is to practice regularly, immerse yourself in the language, and use a variety of resources like books, apps, and language classes. It's also helpful to practice with native speakers."
        },
        {
            "from": "User1",
            "value": "Thank you for the advice."
        },
        {
            "from": "Assistant",
            "value": "You're welcome! Learning a new language can be a rewarding experience. If you have any more questions, feel free to ask."
        }
    ],
    "chat": "You are Mody, a helpful assistant trained by the modalities team. Answer friendly and informatively to the user's messages.\nUser1: What is the best way to learn a new language?\nAssistant:^The best way to learn a new language is to practice regularly, immerse yourself in the language, and use a variety of resources like books, apps, and language classes. It's also helpful to practice with native speakers.\n°$User1: Thank you for the advice.\nAssistant:^You're welcome! Learning a new language can be a rewarding experience. If you have any more questions, feel free to ask.\n°$"
}
```

Run the `apply_chat_template` entry point with:
```bash
modalities data apply_chat_template --config_file_path config_files/data_preparation/apply_chat_template_config.yaml
```

This will create two files
1. The new JSONL file with a new attribute `chat` containing the conversations e.g. `lorem_ipsum_sft_40e0699/lorem_ipsum_sft_converted.40e0699.jsonl`
2. The config used to generate the `chat` e.g. `lorem_ipsum_sft_40e0699/sft_chat_template_config.40e0699.yaml`

> Both files names contain the first 7 symbols of the hash of the config file, to group files which belong together!
> Also, a new directory with the original dataset file name and the hash in it its name is created.

### Create idx and pbin files
Before continuing with the instruction-tuning you need to index the created JSONL and convert it to a tokenized binary file.

> Make sure to use the same hash for correct grouping when defining the output file names!

For example:
```bash
# create idx file
modalities data create_raw_index --index_path data/lorem_ipsum_sft_40e0699/lorem_ipsum_sft_converted.40e0699.idx data/lorem_ipsum_sft_40e0699/lorem_ipsum_sft_converted.40e0699.jsonl 

# create pbin file
modalities  data pack_encoded_data --config_file_path config_files/data_preparation/packed_chat_dataset_config.yaml
```

> The [packed_chat_dataset_config.yaml](config_files/data_preparation/packed_chat_dataset_config.yaml) must use truncation and padding!

In summary, the automatically created folder for all files related to the instruction-tuning data, should look as follows (the hash value might be different depending on your intial apply chat template configuration file):

> lorem_ipsum_sft_40e0699
> ├── lorem_ipsum_sft_converted.40e0699.idx
> ├── lorem_ipsum_sft_converted.40e0699.jsonl
> ├── lorem_ipsum_sft_converted.40e0699.pbin
> ├── packed_chat_dataset_config.40e0699.yaml
> └── sft_chat_template_config.40e0699.yaml

### Instruction-Tuning

With your prepared instruction-tuning data as pbin file, you can now instruction-tune.

Make sure to use the wrapped collate function.

* You need to look up the `b_include_to_loss_token` and `e_include_to_loss_token` as defined within your `sft_chat_template_config.40e0699.yaml`. If configured the pbin creation correctly, you only need to check for matching hash suffixes.
* Set the `loss_ignore_index` which gets ignored by your loss function. In torch this is usually -100.
* We need a tokenizer to tokenize the `b_include_to_loss_token` and `e_include_to_loss_token`
* We need to not re-use the last token

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

with

```yaml
train_dataset:
  component_key: dataset
  variant_key: packed_mem_map_dataset_continuous
  config:
    raw_data_path: ./data/lorem_ipsum_sft_40e0699/lorem_ipsum_sft_converted.40e0699.pbin
    sequence_length: ${settings.training.sequence_length}
    sample_key:  ${settings.referencing_keys.sample_key}
    reuse_last_target: true
```

and with

```yaml
tokenizer:
  component_key: tokenizer
  variant_key: pretrained_hf_tokenizer
  config:
    pretrained_model_name_or_path: data/tokenizer/hf_gpt2
    padding: false
    truncation: false
    special_tokens:
      additional_special_tokens: 
        - "^"
        - "$"
```

Finally, run the instruction-tuning with the `run` entry point:
```bash
torch.distributed.run --nnodes 1 --nproc_per_node 2 --rdzv-endpoint=0.0.0.0:29555 src/modalities/__main__.py run --config_file_path config_files/training/config_lorem_ipsum_sft.yaml
```

> Note, that it is advised to add a special token (which is already known as non-special token to the tokenizers' voabulary) to indicate the end of an assistant turn within the `b_include_to_loss_token` and `e_include_to_loss_token` in your chat template. Change your chat template accordingly and make sure to inlcude this token as special token in the tokenizer configuration for the pbin file creation step and model training!

#### A Note on Tokanization in Huggingface
The special tokens are added to a [Trie](https://en.wikipedia.org/wiki/Trie), so that longer special tokens are split first and then shorter special tokens.
Example from the huggingface documentation:

```python
>>> trie = Trie()
>>> trie.split("[CLS] This is a extra_id_100")
["[CLS] This is a extra_id_100"]

>>> trie.add("[CLS]")
>>> trie.add("extra_id_1")
>>> trie.add("extra_id_100")
>>> trie.split("[CLS] This is a extra_id_100")
["[CLS]", " This is a ", "extra_id_100"]
```

When we add a special token, which exists within the tokenizer voabulary already, HF only marks it as special token (adds it to the trie).
This means, if the sequence we add as special token already exists in the vocab, there is no need to resize the embedding matrix!

## Low-rank Adaption (LorA)

TBD
