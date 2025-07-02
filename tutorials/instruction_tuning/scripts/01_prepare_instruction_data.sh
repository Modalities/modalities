#!/bin/bash

python src/modalities/__main__.py \
  data \
  prepare_instruction_tuning_data \
  --config_file_path tutorials/instruction_tuning/configs/apply_chat_template_config.yaml
