import tempfile
from pathlib import Path

import numpy as np
import pytest

from modalities.api import FileExistencePolicy, create_shuffled_dataset_chunk
from modalities.dataloader.dataset import PackedMemMapDatasetBase
from modalities.dataloader.preprocessing.tokenization.tokenized_file_writer import TokenizedFileWriter


@pytest.fixture
def files_num_documents() -> list[int]:
    return [10, 90, 900]


@pytest.fixture
def pbin_file_path_list(files_num_documents: list[int]) -> list[Path]:
    num_tokens_per_document = 5
    # we can just take the sum of the number of documents to get the vocab size
    # as each document contains a unique token repeated multiple times
    vocab_size = sum(files_num_documents)

    with tempfile.TemporaryDirectory() as temp_dir:
        pbin_file_path_list = []
        offset = 0
        for num_docs in files_num_documents:
            tokenized_dataset = (
                np.arange(offset, num_docs + offset)
                .repeat(num_tokens_per_document)
                .reshape(num_docs, num_tokens_per_document)
            )
            offset += num_docs

            pbin_file_path = Path(temp_dir) / f"test_{num_docs}.pbin"

            token_size_in_bytes = TokenizedFileWriter.get_required_num_of_bytes_to_repr(vocab_size)
            TokenizedFileWriter.write_tokenized_dataset(
                tokenized_dataset=tokenized_dataset,
                tokenized_dataset_file_path=pbin_file_path,
                token_size_in_bytes=token_size_in_bytes,
            )

            pbin_file_path_list.append(pbin_file_path)
        yield pbin_file_path_list


@pytest.mark.parametrize(
    "num_chunks, global_seed, expect_error",
    [
        (900, 1, False),
        (901, 1, True),
        (5000, 1, True),
        (5, 1, False),
        (1, 1, False),
    ],
)
def test_create_shuffled_dataset_chunk(
    pbin_file_path_list: list[Path],
    num_chunks: int,
    global_seed: int,
    expect_error: bool,
):
    def create_chunks(
        num_chunks: int,
        pbin_file_path_list: list[Path],
    ) -> list[np.ndarray]:
        chunks = []
        parent_dir = pbin_file_path_list[0].parent
        for chunk_id in range(num_chunks):
            chunk_file_path = parent_dir / f"chunk_{chunk_id}.pbin"
            create_shuffled_dataset_chunk(
                file_path_list=pbin_file_path_list,
                output_chunk_file_path=chunk_file_path,
                chunk_id=chunk_id,
                num_chunks=num_chunks,
                file_existence_policy=FileExistencePolicy.ERROR,
                global_seed=global_seed,
            )
            dataset = PackedMemMapDatasetBase(raw_data_path=chunk_file_path, sample_key="text", load_index=True)
            tokenized_dataset = dataset[:]["text"]
            chunks.append(tokenized_dataset)
        return chunks

    if expect_error:
        with pytest.raises(ValueError):
            create_chunks(num_chunks, pbin_file_path_list)
        return
    chunks = create_chunks(num_chunks, pbin_file_path_list)

    chunks_combined = []
    for i in range(num_chunks):
        chunks_combined.extend(chunks[i])

    # load the original pbin files
    tokenized_datasets = []
    for pbin_file_path in pbin_file_path_list:
        dataset = PackedMemMapDatasetBase(raw_data_path=pbin_file_path, sample_key="text", load_index=True)
        tokenized_dataset = dataset[:]["text"]
        tokenized_datasets.extend(tokenized_dataset)

    # check that the sorted chunks are equivalent to the original pbin files
    sorted_combined_chunks = list(sorted(chunks_combined, key=lambda x: x[0]))
    sorted_dataset = list(sorted(tokenized_datasets, key=lambda x: x[0]))
    for i in range(len(sorted_combined_chunks)):
        assert all(sorted_combined_chunks[i] == sorted_dataset[i])

    # check that the unsorted chunkds are not equivalent to the original pbin files
    num_documents_equivalent = 0
    for i in range(len(chunks_combined)):
        if len(chunks_combined[i]) == len(tokenized_datasets[i]):
            num_documents_equivalent += int(all(chunks_combined[i] == tokenized_datasets[i]))

    assert num_documents_equivalent < len(chunks_combined)
