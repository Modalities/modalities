import pickle
from multiprocessing import Queue

from modalities.dataloader.create_packed_data import EmbeddedStreamData
from modalities.dataloader.shuffle_tokenized_data import _process_batch, _writer_process, shuffle_tokenized_data


def test_process_batch_with_embedded_stream_with_memmap(tmp_path):
    # Create a temporary file
    file_path = tmp_path / "test_data.pbin"
    data = b"IloveModalities"  # Example data

    with open(file_path, "wb") as f:
        f.write(data)

    # Load the data into memory
    with open(file_path, "rb") as f:
        in_memory_data = f.read()

    # Define a batch
    batch = [(0, 1), (1, 4), (5, 10)]

    # Call the function
    result = _process_batch(batch, in_memory_data)

    # Validate the result
    expected_data = b"IloveModalities"
    expected_lengths = [1, 4, 10]
    assert result == (expected_data, expected_lengths)


def test_writer_process(tmp_path):
    # Example header data
    header_data = b"header"

    # Example queue items
    queue = Queue()
    queue.put((b"data1", [5]))
    queue.put((b"data2", [5]))
    queue.put(None)  # Sentinel

    # Output path
    output_path = tmp_path / "output.pbin"

    # Call the writer process
    _writer_process(output_path, queue, header_data)

    # Validate the result
    with output_path.open("rb") as f:
        written_data = f.read()

    # Expected content: header + data1 + data2 + final index
    expected_index = [(0, 5), (5, 5)]
    expected_content = b"headerdata1data2" + pickle.dumps(expected_index)
    for i in range(len(written_data)):
        print(written_data[i], expected_content[i])
    assert written_data == expected_content


def test_shuffle_tokenized_data(tmp_path):
    # Create test input data
    data = b"IloveModalities"
    data_section_length_as_bytes = len(data).to_bytes(
        EmbeddedStreamData.DATA_SECTION_LENGTH_IN_BYTES, byteorder="little"
    )
    token_size_in_bytes = 4
    token_size_as_bytes = token_size_in_bytes.to_bytes(
        EmbeddedStreamData.TOKEN_SIZE_DESCRIPTOR_LENGTH_IN_BYTES, byteorder="little"
    )
    index = [(0, 1), (1, 4), (5, 10)]

    # Prepare the input file
    input_path = tmp_path / "input.pbin"
    with input_path.open("wb") as f:
        f.write(data_section_length_as_bytes)
        f.write(token_size_as_bytes)
        f.write(data)
        f.write(pickle.dumps(index))

    # Call shuffle_tokenized_data
    output_path = tmp_path / "input_shuffled.pbin"
    shuffle_tokenized_data(input_path, batch_size=1)

    # Validate the output
    assert output_path.is_file()

    with output_path.open("rb") as f:
        # Validate header and data
        data_section_length_as_bytes = f.read(len(data_section_length_as_bytes))
        assert data_section_length_as_bytes == data_section_length_as_bytes
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
