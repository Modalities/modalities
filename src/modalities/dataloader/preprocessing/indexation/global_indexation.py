import pickle
from pathlib import Path

import numpy as np
import tqdm


def _get_global_index_file_path(global_index_root_path: Path) -> Path:
    global_index_file_path = global_index_root_path / f"{global_index_root_path.name}_inorder.idx"
    return global_index_file_path


def _get_file_list(file_list_path: Path) -> list[Path]:
    file_list: list[Path] = []
    with open(file_list_path, "r") as f:
        for line in f:
            file_list.append(Path(line.strip()))
    return file_list


def _get_file_id_file_path_mappings(file_list: list[Path]) -> tuple[dict[Path, int], dict[int, Path]]:
    file_path_to_id = {file_path.with_suffix(""): i for i, file_path in enumerate(file_list)}
    id_to_file_path = {i: file_path.with_suffix("") for i, file_path in enumerate(file_list)}
    return file_path_to_id, id_to_file_path


def _get_local_index_paths(file_list: list[Path], root_index_path: Path, global_index_root_path: Path) -> list[Path]:
    local_index_paths = [
        path.with_suffix(".idx")
        for path in file_list
        if (root_index_path / path).is_relative_to(global_index_root_path)
    ]
    return local_index_paths


def _get_total_num_documents(local_index_paths: list[Path], root_index_path: Path) -> int:
    num_documents = 0
    for local_index_path in tqdm.tqdm(local_index_paths, desc="Counting total number of documents"):
        with open(root_index_path / local_index_path, "rb") as f:
            index = pickle.load(f)
            num_documents += len(index)
    return num_documents


def _populate_global_index_array(
    global_index_file_path: Path,
    num_documents: int,
    local_index_paths: list[Path],
    root_index_path: Path,
    file_path_to_id: dict[Path, int],
) -> np.memmap:
    shape = (num_documents + 1, 3)
    global_index_array = np.memmap(global_index_file_path, dtype="int64", mode="w+", shape=shape)

    # the first row is reserved for the shape of the array and whether rows are shuffled.
    # <num rows, num columns, is_shuffled>
    global_index_array[0] = np.array([*shape, 0])
    start_index = 1
    for local_index_path in tqdm.tqdm(local_index_paths, desc="Populating global index array"):
        with open(root_index_path / local_index_path, "rb") as f:
            local_index = pickle.load(f)

        local_index_array = np.array(local_index)
        # add the file id to the local index
        file_id = file_path_to_id[local_index_path.with_suffix("")]
        local_index_array = np.insert(local_index_array, 0, file_id, axis=1)

        global_index_array[start_index : start_index + len(local_index_array)] = local_index_array
        start_index += len(local_index_array)
    global_index_array.flush()
    return global_index_array


def create_global_index(file_list_path: Path, root_index_path: Path, global_index_root_path: Path) -> Path:
    global_index_file_path = _get_global_index_file_path(global_index_root_path)

    file_list = _get_file_list(file_list_path)

    file_path_to_id, _ = _get_file_id_file_path_mappings(file_list)
    local_index_paths = _get_local_index_paths(file_list, root_index_path, global_index_root_path)
    num_documents = _get_total_num_documents(local_index_paths, root_index_path)

    _populate_global_index_array(
        global_index_file_path, num_documents, local_index_paths, root_index_path, file_path_to_id
    )
    return global_index_file_path


def create_shuffled_global_index(global_index_file_path: Path) -> Path:
    global_shuffled_index_file_path = (
        global_index_file_path.parent / f"{global_index_file_path.stem.replace('inorder', 'shuffle_index')}.idx"
    )
    print(global_shuffled_index_file_path)

    # global index array
    num_rows, _, _ = np.memmap(global_index_file_path, dtype="int64", mode="r")[0:3]

    print(f"Shuffling {num_rows-1} global index indices")
    # we count from 1 since the 0th row contains meta information (num_rows, num_cols, is_shuffled)
    indices = np.arange(1, num_rows)
    np.random.shuffle(indices)

    print(f"Writing out shuffled global index array with {num_rows} elements")
    global_shuffled_index_array = np.memmap(
        global_shuffled_index_file_path, dtype="int64", mode="w+", shape=(len(indices),)
    )
    chunk_size = 10
    for i in tqdm.tqdm(range(0, len(indices), chunk_size)):
        chunk_indices = indices[i : i + chunk_size]
        global_shuffled_index_array[i : i + len(chunk_indices)] = chunk_indices
    global_shuffled_index_array.flush()
    return global_shuffled_index_file_path
