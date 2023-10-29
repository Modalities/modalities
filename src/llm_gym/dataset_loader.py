from torch.utils.data import Sampler, Dataset
from torch.utils.data.dataloader import DataLoader, _collate_fn_t, _worker_init_fn_t, T_co
from typing import Iterable, List, Optional, Union


class LLMDataLoader(DataLoader[T_co]):
    def __init__(self, dataset_tag: str, dataset: Dataset[T_co], batch_size: Optional[int] = 1,
                 shuffle: Optional[bool] = None, sampler: Union[Sampler, Iterable, None] = None,
                 batch_sampler: Union[Sampler[List], Iterable[List], None] = None,
                 num_workers: int = 0, collate_fn: Optional[_collate_fn_t] = None,
                 pin_memory: bool = False, drop_last: bool = False,
                 timeout: float = 0, worker_init_fn: Optional[_worker_init_fn_t] = None,
                 multiprocessing_context=None, generator=None,
                 *, prefetch_factor: Optional[int] = None,
                 persistent_workers: bool = False,
                 pin_memory_device: str = ""):
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                         batch_sampler=batch_sampler, num_workers=num_workers, collate_fn=collate_fn,
                         pin_memory=pin_memory, drop_last=drop_last, timeout=timeout, worker_init_fn=worker_init_fn,
                         multiprocessing_context=multiprocessing_context, generator=generator, prefetch_factor=prefetch_factor,
                         persistent_workers=persistent_workers, pin_memory_device=pin_memory_device)

        self._dataset_tag = dataset_tag

    @property
    def dataset_tag(self) -> str:
        return self._dataset_tag
