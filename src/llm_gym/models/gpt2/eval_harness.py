
# TODO Remove this file before merging branch into main
import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from llm_gym.models.gpt2.gpt2_model import GPTConfig, AttentionConfig, Attention, Activation
from llm_gym.models.gpt2.gpt2_evaluation import PretrainedGPTModel, PretrainedGPTConfig


attention_config = AttentionConfig(attention_type=Attention("default_attention"), scaling_factor=3)

config = GPTConfig(
    block_size=1024,
    vocab_size=50257,
    n_layer=4,
    n_head=16,
    n_embd=1024,
    dropout=0.01,
    bias=True,
    attention=attention_config,
    activation=Activation.GELU,
    epsilon=1e-5
)
pretrained_config = PretrainedGPTConfig(config=config)
model = PretrainedGPTModel(
    prediction_publication_key="logits",
    config=pretrained_config
)

model.save_pretrained("/home/richard-rutmann/s3/models/llm_gym/test")

# batch size, sequence length, embedding dimensionality (n_embd)
test_tensor = torch.randint(10, size=(5, 10))
model = model.eval()
output_before_loading = model.forward(test_tensor)

AutoConfig.register("llm_gym_gpt2", PretrainedGPTConfig)
AutoModelForCausalLM.register(PretrainedGPTConfig, PretrainedGPTModel)
model.push_to_hub("llm_gym_gpt2", token="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")


config = PretrainedGPTConfig.from_pretrained("/home/richard-rutmann/s3/models/llm_gym/test/config.json")
# config = AutoConfig.from_pretrained("rrutmann/llm_gym_gpt2")
loaded_model = AutoModelForCausalLM.from_pretrained("rrutmann/llm_gym_gpt2", prediction_publication_key="logits", config=config)

loaded_model = loaded_model.eval()
loaded_model.forward(test_tensor)
output_after_loading = loaded_model.forward(test_tensor)

# assert output_after_loading == output_before_loading
#
# for p in loaded_model.parameters():
#     print(p)
#     break
#
# for p in model.parameters():
#     print(p)
#     break