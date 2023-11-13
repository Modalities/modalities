
from llm_gym.models.gpt2.gpt2_model import GPTConfig, AttentionConfig, Attention, Activation
from llm_gym.models.gpt2.gpt2_evaluation import PretrainedGPTModel, PretrainedGPTConfig


attention_config = AttentionConfig(attention_type=Attention("default_attention"), scaling_factor=2)

config = GPTConfig(
    block_size=10,
    vocab_size=128,
    n_layer=4,
    n_head=12,
    n_embd=120,
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
