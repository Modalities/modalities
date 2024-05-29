num_bpe_operations=$1
bpecodes_file_suffix=$2_${num_bpe_operations}

subword-nmt learn-bpe -s $num_bpe_operations < training.txt > bpecodes_${bpecodes_file_suffix}

python src/modalities/__main__.py data get_coca_tokenizer_and_vocab\
    bpecodes_${bpecodes_file_suffix}\
    $bpecodes_file_suffix