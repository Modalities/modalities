def test_create_packed_dataset(indexed_dummy_data_path, gpt2_tokenizer):
    pass
    # block_size = 5
    # packed_generator = SFTMemMapDataset(
    #     src_path=indexed_dummy_data_path.raw_data_path, tokenizer=gpt2_tokenizer, number_of_processes=2
    # )
    # default_packed_dataset_path = packed_generator._default_destination_path()
    # assert not default_packed_dataset_path.is_file()
    # packed_generator.run()
    # packed_dataset = PackedMemMapDatasetContinuous(
    #     default_packed_dataset_path, block_size=block_size, sample_key="input_ids"
    # )

    # start_of_jsonl_content = "0 Lorem ipsum dolor sit amet, consetetur sadipscing elitr"
    # tokenized_start_of_jsonl_content = gpt2_tokenizer(start_of_jsonl_content)["input_ids"]
    # packed_dataset_iterator = iter(packed_dataset)
    # np.testing.assert_equal(tokenized_start_of_jsonl_content[:block_size], next(packed_dataset_iterator)["input_ids"])
    # np.testing.assert_equal(
    #     tokenized_start_of_jsonl_content[block_size : 2 * block_size], next(packed_dataset_iterator)["input_ids"]
    # )
    # assert len(packed_dataset._embedded_stream_data.index_base) == 12

    # # check validity of index section in packed dataset
    # for idx, (offset, entry_length) in enumerate(packed_dataset._embedded_stream_data.index_base[:-1]):
    #     assert offset + entry_length == packed_dataset._embedded_stream_data.index_base[idx + 1][0]
