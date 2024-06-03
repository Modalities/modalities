import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from modalities.models.mamba.mamba_model import MambaLLM


def load_paper_model():
    model = AutoModelForCausalLM.from_pretrained('Q-bert/Mamba-130M', trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained('Q-bert/Mamba-130M')
    text = "Hi"
    input_ids = tokenizer.encode(text, return_tensors="pt")
    output = model.generate(input_ids, max_length=20, num_beams=5, no_repeat_ngram_size=2)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(generated_text)
    return model


def load_our_model():
    # reference implementation how to load the model from checkpoint: generate_text("config_files/text_generation/text_generation_mamba.yaml")
    with open("config_files/text_generation/text_generation_mamba.yaml") as f:
        component_config_dict = yaml.load(f, Loader=yaml.FullLoader)
        model = MambaLLM(component_config_dict)  # todo put the dict in here
        # load checkpoint
        # return model


if __name__ == '__main__':
    our_model = load_our_model()
    paper_model = load_paper_model
    input = "hi"

    # todo:
    # our_model(input)
    # paper_model(input)


"""

Results 03/06

- adapter eval harness does not work anymore (old PR was still open)
- we can load the mamba model from the paper (130M, trained on 300B tokens)
- our model has the same size, but was only trained on 16M tokens so far --> 30B tokens would be nice, 4 nodes --> 32 GPUs

--> more training needed
--> do over the whole eval harness implementation

"""
