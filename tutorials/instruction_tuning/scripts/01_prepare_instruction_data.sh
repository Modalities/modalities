#!/bin/bash

modalities \
  data \
  prepare_instruction_tuning_data \
  --config_file_path configs/apply_chat_template_config.yaml
