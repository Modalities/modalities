from transformers import PreTrainedTokenizerFast

tokenizer_path = "/raid/s3/opengptx/alexj/llm_gym/models/SmolLM-1.7B/"


if __name__ == '__main__':
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    b_include_to_loss_token = "^"
    e_include_to_loss_token = "$"

    tokenized_output = tokenizer(b_include_to_loss_token)
    print(tokenized_output)

    tokenized_output = tokenizer(e_include_to_loss_token)
    print(tokenized_output)