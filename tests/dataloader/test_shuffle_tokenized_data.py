import pickle

from modalities.dataloader.create_packed_data import EmbeddedStreamData
from modalities.preprocessing.shuffle_data import DataShuffler


def _tokenize(text: str, vocabulary: dict[str, int]) -> list[int]:
    text = text.lower()
    return [vocabulary[char] for char in text]


def _convert_tokens_to_bytes(tokens: list[int], num_bytes_per_token: int) -> bytes:
    return b"".join([token.to_bytes(num_bytes_per_token, byteorder="little", signed=False) for token in tokens])


def test_process_batch(tmp_path, encoding_set_up):
    vocabulary, num_bytes_per_token = encoding_set_up
    # Create a temporary file
    file_path = tmp_path / "test_data.pbin"
    data = _tokenize(text="IloveModalities", vocabulary=vocabulary)
    data = _convert_tokens_to_bytes(data, num_bytes_per_token=num_bytes_per_token)

    with open(file_path, "wb") as f:
        f.write(data)

    # Load the data into memory
    with open(file_path, "rb") as f:
        in_memory_data = f.read()

    # Define a batch
    batch = [(0, 1), (1, 4), (5, 10)]

    # Call the function
    new_data, new_index = DataShuffler._process_batch(batch=batch, data=in_memory_data, start_position=0)

    # Validate the result
    expected_data = data
    expected_index = batch
    assert (new_data, new_index) == (expected_data, expected_index)


def test_shuffle_tokenized_data(tmp_path, encoding_set_up):
    vocabulary, num_bytes_per_token = encoding_set_up
    # Create test input data
    data = _tokenize(text="IloveModalities", vocabulary=vocabulary)
    data = _convert_tokens_to_bytes(data, num_bytes_per_token=num_bytes_per_token)
    data_section_length_as_bytes = len(data).to_bytes(
        EmbeddedStreamData.DATA_SECTION_LENGTH_IN_BYTES, byteorder="little"
    )
    token_size_as_bytes = num_bytes_per_token.to_bytes(
        EmbeddedStreamData.TOKEN_SIZE_DESCRIPTOR_LENGTH_IN_BYTES, byteorder="little"
    )
    index = [(0, 1), (1, 4), (5, 10)]

    # Prepare the input file
    input_path = tmp_path / "input.pbin"
    output_path = tmp_path / "output.pbin"
    with input_path.open("wb") as f:
        f.write(data_section_length_as_bytes)
        f.write(token_size_as_bytes)
        f.write(data)
        f.write(pickle.dumps(index))

    for batch_size in [1, 2, 3]:
        # Call shuffle_tokenized_data
        output_path = tmp_path / "input_shuffled.pbin"
        DataShuffler.shuffle_tokenized_data(
            input_data_path=input_path, output_data_path=output_path, batch_size=batch_size
        )

        # Validate the output
        assert output_path.is_file()

        with output_path.open("rb") as f:
            # Validate header and data
            data_section_length_as_bytes_written = f.read(EmbeddedStreamData.DATA_SECTION_LENGTH_IN_BYTES)
            assert data_section_length_as_bytes_written == data_section_length_as_bytes
            assert f.read(len(token_size_as_bytes)) == token_size_as_bytes
            data_len = int.from_bytes(data_section_length_as_bytes, byteorder="little")
            data_written = f.read(data_len)

            # Validate the shuffled index
            written_index = pickle.loads(f.read())

            # Extract substrings from the data using written_index
            extracted_substrings = [data_written[start : start + length] for start, length in written_index]

            # Verify that these substrings match the original defined ones
            original_substrings = [data[start : start + length] for start, length in index]

            # Ensure that extracted substrings are a valid permutation of original substrings
            assert sorted(extracted_substrings) == sorted(original_substrings)
