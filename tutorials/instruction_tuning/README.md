# Instruction Tuning with Modalities 🚀

This tutorial guides you through fine-tuning a base language model to follow instructions and act as a helpful chat assistant.

We'll cover three main steps:

1.  **📝 Data Preparation:** Convert a standard instruction dataset into a tokenized format suitable for training, using a custom chat template.
2.  **🧠 Model Fine-Tuning:** Use the `modalities` library to instruction-tune the Qwen/Qwen2.5-0.5B model.
3.  **💬 Text Generation:** Interact with your newly fine-tuned model to see its conversational abilities.

-----

## What is Instruction Tuning?

Instruction tuning is the process of fine-tuning a pre-trained language model on conversational data. The goal is to teach the model how to follow instructions and respond as a helpful assistant.

To do this, we only want the model to learn the assistant's speaking style, not the user's. We achieve this through **loss masking**, where we calculate the learning loss only on the assistant's responses.

For example, in the conversation below, the model's weights are only updated based on the text that isn't struck through:

> ~~You are Mody, a helpful assistant trained by the modalities team. Answer friendly and informatively to the user's messages.\\nUser: What is the best way to learn a new language?\\nAssistant:<start_assistant>~~
> The best way to learn a new language is to practice regularly, immerse yourself in the language, and use a variety of resources like books, apps, and language classes. It's also helpful to practice with native speakers.\\n°
> ~~User: Thank you for the advice.\\nAssistant:<start_assistant>~~
> You're welcome\! Learning a new language can be a rewarding experience. If you have any more questions, feel free to ask.\\n°
> ~~<end_assistant>~~

> **Note:** This tutorial currently supports fast and slow Hugging Face tokenizers, as special tokens must be added to the tokenizer for loss masking.

-----

## Step 1: Prepare the Instruction-Tuning Data 📝

We'll use a custom script to format our conversational data. This script applies a chat template to each conversation, tokenizes the text, and splits it into `train`, `test`, and `validation` sets.

See the configuration files for more details:

  * [`apply_chat_template_config.yaml`](configs/apply_chat_template_config.yaml)
  * [`packed_chat_dataset_config.yaml`](configs/packed_chat_dataset_config.yaml)

To prepare the data, run the `prepare_instruction_tuning_data` entry point using the provided script:

```bash
bash scripts/prepare_instruction_data.sh
```

### How Data Preparation Works

The script uses the settings in your configuration file to:

  * **Load Data**: Reads a JSONL file where each line is a conversation.
  * **Apply Chat Template**: Uses a [Jinja2](https://jinja.palletsprojects.com/en/3.1.x/) template to structure the raw data into a chat format. We use special tokens, `b_include_to_loss_token` and `e_include_to_loss_token`, to mark the beginning and end of the assistant's messages. These markers tell the model which parts of the text to learn from.
    * **Map Roles**: Converts role identifiers (e.g., `"role": "user"`) into display names (e.g., `User:`).
  * **Tokenize and Save**: Tokenizes the formatted text and writes it as binary (`.pbin`) and index files (`.idx`) for efficient loading during training.

> **⚠️ Important Limitation:** Currently, special tokens like `b_include_to_loss_token` must already exist in the tokenizer's vocabulary. We cannot add new tokens on the fly because resizing the model's embedding matrix is not yet supported. See the corresponding [issue](https://github.com/Modalities/modalities/issues/208) for more details.

### Output Files

After running the script, you will find a new directory (`prepared_data/instruction_tuning_data_8820ad4`) containing:

  * **JSONL files**: The split datasets with the applied chat template (e.g., `instruction_tuning_data_applied_chat_template_train.8820ad4.jsonl`).
  * **Binary and Index files**: The tokenized data partitions for training (e.g., `..._train.8820ad4.pbin` and `..._train.8820ad4.idx`).
  * **Config files**: The configuration files used for data generation, saved for reproducibility.

All generated files share a unique hash (e.g., `8820ad4`) based on the config file, making it easy to group them.

-----

## Step 2: Fine-Tune the Model 🧠

With your data prepared, you can now fine-tune the model. We use a collate function wrapper, `mask_loss_collator_wrapper`, to handle the loss masking during training.

See the full configuration in [train_instruct_model_fsdp1_config.yaml](configs/train_instruct_model_fsdp1_config.yaml).

### Key Configuration Changes

The core of the instruction-tuning setup lies in the `collate_fn` and `train_dataset` configurations:

1.  **`collate_fn`**: This function prepares data batches for the model.

      * It uses the special tokens (`b_include_to_loss_token` and `e_include_to_loss_token`) to identify the assistant's replies.
      * It masks the loss for all other tokens by setting their label to `-100`, which is the standard ignore index in PyTorch.
      * It requires a tokenizer configuration that recognizes your special tokens.

2.  **`train_dataset`**:

      * Set `reuse_last_target: false`. This is **crucial** to load the truncated and padded data correctly.

Finally, start the fine-tuning process by running:

```bash
bash scripts/train_instruction_tuning_model.sh
```

-----

## Step 3: Chat with Your Fine-Tuned Model 💬

Once training is complete, it's time to chat with your model\!

1.  Update the model path in [`text_generation_config.yaml`](configs/text_generation_config.yaml) to point to the checkpoint you just created in `tutorials/instruction_tuning/checkpoints`.
2.  Run the generation script:
    ```bash
    bash scripts/03_generate_text.sh
    ```
3.  When prompted, ask a question like, "What is 2 + 2?".

Since we are using completion mode, the script prefixes your input with the start of the chat template. The model will generate a response and stop automatically because we defined an end token (`<|endoftext|>`) during data preparation.

> **Note:** Training on a small dataset (1k examples) for a single epoch is not enough to achieve high performance, but it's a great way to verify that your pipeline is working correctly\!
