import multiprocessing as mp
import queue
import traceback
from multiprocessing.synchronize import Event
from typing import Any, Optional

from modalities.dataloader.preprocessing.queued_processing.processing_strategy_if import ProcessingStrategyIF
from modalities.exceptions import ProcessingStrategyDoneException, ProcessorException, ProcessorStopEventException
from modalities.utils.logging import get_logger


class QueueConsumer:
    def __init__(self, in_q: mp.Queue, in_q_timeout: int):
        self._in_q = in_q
        self._in_q_timeout = in_q_timeout
        self._consumed_items = 0

    def get_item(self, stop_event: Event) -> Any:
        while not stop_event.is_set():
            try:
                item = self._in_q.get(timeout=self._in_q_timeout)
            except queue.Empty:
                continue
            if item is None:
                pass
            self._consumed_items += 1
            return item
        raise ProcessorStopEventException("Stop event was set")


class QueueProducer:
    def __init__(self, out_q: mp.Queue, out_q_timeout: int):
        self._out_q = out_q
        self._out_q_timeout = out_q_timeout

    def put_item(self, item: Any, stop_event: Event):
        while not stop_event.is_set():
            try:
                self._out_q.put(item, timeout=self._out_q_timeout)
            except queue.Full:
                continue
            return
        raise ProcessorStopEventException("Stop event was set")


class Processor(mp.Process):
    def __init__(
        self,
        out_qs: dict[str, mp.Queue],
        in_q_timeout: int,
        out_q_timeout: int,
        strategy: ProcessingStrategyIF,
        process_id: str,
        process_type: str,
        stop_event: Event,
        set_stop_event_on_processing_error: bool,
        in_q: mp.Queue = None,
        logging_message_q_key: Optional[str] = None,
    ):
        super().__init__()

        self._consumer = QueueConsumer(in_q, in_q_timeout) if in_q is not None else None
        self._producers: dict[str, QueueProducer] = {
            q_key: QueueProducer(out_q, out_q_timeout) for q_key, out_q in out_qs.items()
        }
        self._strategy = strategy
        self._stop_event = stop_event
        self._process_type = process_type
        self._process_id = process_id
        self.exit_on_processing_error = set_stop_event_on_processing_error
        self._logging_message_q_key = logging_message_q_key
        # if the consumer is None, we are the first processor in the pipeline and we need to generate the items
        self._processing_fun = self._generate_item if self._consumer is None else self._process_item

    @property
    def process_id(self) -> str:
        return self._process_id

    @property
    def process_type(self) -> str:
        return self._process_type

    @property
    def full_name(self) -> str:
        return f"{self._process_type}:{self._process_id}"

    def _generate_item(self):
        try:
            processed_sub_items: dict[str, Any] = self._strategy.process()
        except ProcessingStrategyDoneException as e:
            self._strategy.finalize()
            get_logger().info(f"{self.full_name} received done (iterator exhausted). Exiting...")
            raise e
        self._forward_sub_items(processed_sub_items)

    def _process_item(self):
        item = self._consumer.get_item(stop_event=self._stop_event)
        if item is None:
            self._strategy.finalize()
            raise ProcessingStrategyDoneException(f"{self.full_name} received done (poison pill).")
        # process the item
        try:
            processed_sub_items: dict[str, Any] | None = self._strategy.process(item)
        except Exception as e:
            get_logger().error(f"{self.full_name} failed to process item {item}. Error: {e}")
            if self.exit_on_processing_error:
                raise ProcessorException(f"{self.full_name} failed to process item {item}.") from e
            return  # continue with the next item
        # forward the processed sub items to the respective queues
        self._forward_sub_items(processed_sub_items)

    def _forward_sub_items(self, processed_sub_items: dict[str, Any]):
        # place the processed sub items in the correct out queues
        for destination_q_key, processed_sub_item in processed_sub_items.items():
            if destination_q_key == self._logging_message_q_key:
                processed_sub_item.process_id = self._process_id
                processed_sub_item.process_type = self._process_type
            self._producers[destination_q_key].put_item(processed_sub_item, stop_event=self._stop_event)

    def run(self):
        try:
            with self._strategy:
                while True:
                    self._processing_fun()
        except ProcessingStrategyDoneException:
            pass
        except ProcessorStopEventException:
            # if the stop event was set, some process in the pipeline failed and we need to exit
            get_logger().info(f"{self.full_name} received forced stop event. Exiting...")
        except Exception as e:
            # in this block, every exception comes from this very process and we need to set the stop event
            # to signal the other processes of the pipeline that something went wrong
            stacktrace = traceback.format_exc()
            get_logger().error(f"Stacktrace for {self.full_name} : {stacktrace}")
            get_logger().error(f"{self.full_name} failed with error: {e}, setting stop event")
            self._stop_event.set()
        get_logger().error(f"{self.full_name} exiting...")
