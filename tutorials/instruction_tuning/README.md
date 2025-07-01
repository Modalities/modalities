# Instruction Tuning with Modalities

> Instruction-tuning currently only works with fast and slow Huggingface tokenizers, as the special tokens need to be added to the tokenizer.

The goal of instruction-tuning is to let the model learn instruction-following capabilites, so that it acts as an helpful assistant within an chat environment.
For this, we need to further fine-tune the model on conversational instruction data.
Specifically, we need the model to only learn to act as the assistant. Hence, we allow full attention on conversation, but calculate the loss only on the assistants untterances. 

For example, we only calculate the loss for the words not struck-trough:

> ~~You are Mody, a helpful assistant trained by the modalities team. Answer friendly and informatively to the user's messages.\nUser1: What is the best way to learn a new language?\nAssistant:^~~
> The best way to learn a new language is to practice regularly, immerse yourself in the language, and use a variety of resources like books, apps, and language classes. It's also helpful to practice with native speakers.\n°
> ~~$User1: Thank you for the advice.\nAssistant:^~~
> You're welcome! Learning a new language can be a rewarding experience. If you have any more questions, feel free to ask.\n°
> ~~$~~

### Overview

To prepare the instruction-tuning data we created a new entry point `prepare_instruction_tuning_data`, which requires a [configuration file](./config_files/data_preparation/apply_chat_template_config.yaml). Within it we define:
* The path to instruction-tuning dataset as a JSONL file wereas each line contains a structured conversation as an array of dictionaries (configured by the yaml entry: `messages_key: messages`).
* A [jinja2](https://jinja.palletsprojects.com/en/3.1.x/) chat template which defines the rules how to glue `chat_template_data` and the data within the JSONL together to one `chat` string.
  * As part of the `chat_template_data`, we require the special tokens `b_include_to_loss_token` and `e_include_to_loss_token`. A special, required `chat_template_data` is `messages` wich allows to loop over user/assistant turns of one example conversation (i.e. the data in the `messages_key` column)
* Information how to split the created dataset

> Note: The special tokens `b_include_to_loss_token` and `e_include_to_loss_token` should be tokens already present in the tokenizers vocabulary. They will be marked as special tokens for correct tokenization and loss masking. Once resizing the embedding matrix is supported, this is not necessary anymore.

##### Example

Input JSONL file entry:
```json
{
    "id": 16,
    "messages": [
        {
            "role": "human_1",
            "content": "What is the best way to learn a new language?"
        },
        {
            "role": "gpt",
            "content": "The best way to learn a new language is to practice regularly, immerse yourself in the language, and use a variety of resources like books, apps, and language classes. It's also helpful to practice with native speakers."
        },
        {
            "role": "human_1",
            "content": "Thank you for the advice."
        },
        {
            "role": "gpt",
            "content": "You're welcome! Learning a new language can be a rewarding experience. If you have any more questions, feel free to ask."
        }
    ]
}
```

Config: See [](configs/apply_chat_template_config.yaml) and [](configs/packed_chat_dataset_config.yaml)

Created JSONL file entry:
```json
{
    "id": 16,
    "messages": [
        {
            "role": "User1",
            "content": "What is the best way to learn a new language?"
        },
        {
            "role": "Assistant",
            "content": "The best way to learn a new language is to practice regularly, immerse yourself in the language, and use a variety of resources like books, apps, and language classes. It's also helpful to practice with native speakers."
        },
        {
            "role": "User1",
            "content": "Thank you for the advice."
        },
        {
            "role": "Assistant",
            "content": "You're welcome! Learning a new language can be a rewarding experience. If you have any more questions, feel free to ask."
        }
    ],
    "chat": "You are Mody, a helpful assistant trained by the modalities team. Answer friendly and informatively to the user's messages.\nUser1: What is the best way to learn a new language?\nAssistant:^The best way to learn a new language is to practice regularly, immerse yourself in the language, and use a variety of resources like books, apps, and language classes. It's also helpful to practice with native speakers.\n°$User1: Thank you for the advice.\nAssistant:^You're welcome! Learning a new language can be a rewarding experience. If you have any more questions, feel free to ask.\n°$"
}
```

### Prepare Instruction-tuning Data

> **Limitation:** 
> Currently only tokens already known to the tokenizers vocabulary can be added, as resizing the embedding matrix is not yet supported!
> See the corresponding [issue](https://github.com/Modalities/modalities/issues/208).


Run the `prepare_instruction_tuning_data` entry point with: [](scripts/prepare_instruction_data.sh)


This will create / copy the following files:

```
tutorials/instruction_tuning/prepared_data
└── instruction_tuning_data_8820ad4
    ├── instruction_tuning_data_applied_chat_template_test.8820ad4.idx
    ├── instruction_tuning_data_applied_chat_template_test.8820ad4.jsonl
    ├── instruction_tuning_data_applied_chat_template_test.8820ad4.pbin
    ├── instruction_tuning_data_applied_chat_template_train.8820ad4.idx
    ├── instruction_tuning_data_applied_chat_template_train.8820ad4.jsonl
    ├── instruction_tuning_data_applied_chat_template_train.8820ad4.pbin
    ├── instruction_tuning_data_applied_chat_template_val.8820ad4.idx
    ├── instruction_tuning_data_applied_chat_template_val.8820ad4.jsonl
    ├── instruction_tuning_data_applied_chat_template_val.8820ad4.pbin
    ├── pbin_config_test.8820ad4.yaml
    ├── pbin_config_train.8820ad4.yaml
    ├── pbin_config_val.8820ad4.yaml
    └── instruction_chat_template_config.8820ad4.yaml
```

All files names contain the first 7 symbols of the hash of the config file, to group files which belong together!
Also, a new directory with the original dataset file name and the hash in it its name is created.

1. The JSONLs files with a new attribute `chat` containing the conversations, split into train, test, val e.g. `instruction_tuning_data_applied_chat_template_train.8820ad4.jsonl`
2. The config used to generate the `chat` e.g. `instruction_chat_template_config.8820ad4.yaml`
3. The idx and pbin files for each dataset partition e.g. `instruction_tuning_data_applied_chat_template_train.8820ad4.idx` and `instruction_tuning_data_applied_chat_template_train.8820ad4.pbin`
4. The config file used to create the pbin files. For each partition (train, test, val), only the `src_path`, `index_path` and `dst_path` are replaced automatically, the rest remains as in the original pbin creation config file, as pointed to within [](tutorials/instruction_tuning/configs/apply_chat_template_config.yaml): `pbin_creation_config_file_path: tutorials/instruction_tuning/configs/packed_chat_dataset_config.yaml`

> Note: The [packed_chat_dataset_config.yaml](config_files/data_preparation/packed_chat_dataset_config.yaml) must use truncation and padding!


### Instruction-Tuning

With your prepared instruction-tuning data as pbin file, you can now instruction-tune.

Make sure to use the wrapped collate function.

* You need to look up the `b_include_to_loss_token` and `e_include_to_loss_token` as defined within your `sft_chat_template_config.09ca9ed.yaml`.
* Set the `loss_ignore_index` which gets ignored by your loss function. In torch this is usually -100.
* We need a tokenizer to tokenize the `b_include_to_loss_token` and `e_include_to_loss_token`
* We need to not re-use the last token

See [](configs/train_instruct_model_fsdp1_config.yaml) for a full example. Below, the core changes needed for instruction tuning are listed.
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
    raw_data_path: ./data/lorem_ipsum_sft_09ca9ed/lorem_ipsum_sft_converted_train.09ca9ed.pbin
    sequence_length: ${settings.training.sequence_length}
    sample_key:  ${settings.referencing_keys.sample_key}
    reuse_last_target: false
```

and with

```yaml
tokenizer:
  component_key: tokenizer
  variant_key: pretrained_hf_tokenizer
  config:
    pretrained_model_name_or_path: data/tokenizer/hf_gpt2
    padding: max_length
    truncation: true
    max_length: ${settings.sequence_length}
    special_tokens:
      pad_token: ${settings.eod_token}
      additional_special_tokens: 
        - ^
        - $
```

Finally, run the instruction-tuning with the `run` entry point: [](scripts/train_instruction_tuning_model.sh)

> Note, that it is advised to add a special token (which is already known as non-special token to the tokenizer's voabulary) to indicate the end of an assistant turn within the `b_include_to_loss_token` and `e_include_to_loss_token` in your chat template. Change your chat template accordingly and make sure to inlcude this token as special token in the tokenizer configuration for the pbin file creation step and model training! 

#### A Note on Tokenization in Huggingface
The special tokens are added to a [Trie](https://en.wikipedia.org/wiki/Trie).
With that data structure, longer special tokens are matched with a higher priority than shorter ones. Regular tokens are tokenized after handling the special tokens first.
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
