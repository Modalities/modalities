import torch

from modalities.batch import DatasetBatch


def configure_dataloader_mock(
    batch_size: int,
    seq_len: int,
    num_batches: int,
    sample_key: str,
    target_key: str,
    llm_data_loader_mock,
):
    sample_tensor = torch.randint(size=(batch_size, seq_len), low=1, high=100)
    samples = {sample_key: sample_tensor[:, :-1]}
    targets = {target_key: sample_tensor[:, 1:]}

    batches = [DatasetBatch(targets=targets, samples=samples) for _ in range(num_batches)]

    llm_data_loader_mock.__iter__ = lambda _: iter(batches)
    llm_data_loader_mock.batch_size = batch_size
    llm_data_loader_mock.fast_forward_batch_id = 0
    llm_data_loader_mock.__len__ = lambda _: num_batches

    return llm_data_loader_mock, batches
