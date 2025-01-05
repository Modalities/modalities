import multiprocessing as mp
import queue
import traceback
from multiprocessing.synchronize import Event
from typing import Any, Optional

from modalities.dataloader.preprocessing.queued_processing.processing_strategy_if import ProcessingStrategyIF
from modalities.exceptions import ProcessorStopEventException
from modalities.utils.logging import get_logger


class Processor(mp.Process):
    class QueueConsumer:
        def __init__(self, in_q: mp.Queue, in_q_timeout: int):
            self._in_q = in_q
            self._in_q_timeout = in_q_timeout

        def get_item(self, stop_event: Event) -> Any:
            while not stop_event.is_set():
                try:
                    item = self._in_q.get(timeout=self._in_q_timeout)
                except queue.Empty:
                    continue
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

    def __init__(
        self,
        in_q: mp.Queue,
        out_qs: dict[str, mp.Queue],
        in_q_timeout: int,
        out_q_timeout: int,
        strategy: ProcessingStrategyIF,
        process_id: str,
        process_type: str,
        stop_event: Event,
        logging_message_q_key: Optional[str] = None,
    ):
        super().__init__()
        self._consumer = Processor.QueueConsumer(in_q, in_q_timeout)
        self._producers: dict[str, Processor.QueueProducer] = {
            q_key: Processor.QueueProducer(out_q, out_q_timeout) for q_key, out_q in out_qs.items()
        }

        self._strategy = strategy
        self._stop_event = stop_event
        self._process_type = process_type
        self._process_id = process_id
        self._logging_message_q_key = logging_message_q_key

    def run(self):
        with self._strategy:
            while True:
                try:
                    item = self._consumer.get_item(stop_event=self._stop_event)
                except ProcessorStopEventException:
                    get_logger().info(f"{self._process_id} stopped due to forced stop event")
                    break
                if item is None:
                    get_logger().info(f"{self._process_id} received regular poison pill, exiting...")
                    self._strategy.finalize()
                    break
                try:
                    processed_sub_items: dict[str, Any] | None = self._strategy.process(item)
                except Exception as e:
                    get_logger().error(
                        f"{self._process_type}:{self._process_id} failed to process item {item}. Error: {e}"
                    )
                    stacktrace = traceback.format_exc()
                    get_logger().error(f"Stacktrace for {self._process_type}:{self._process_id} : {stacktrace}")
                    get_logger().error(f"{self._process_id} setting stop event and then exiting...")
                    self._stop_event.set()
                    break

                # if the strategy returns None, we don't have to put anything in any of the out_qs
                if processed_sub_items is None:
                    continue
                else:
                    # place the processed  sub items in the correct out queues
                    for destination_q_key, processed_sub_item in processed_sub_items.items():
                        if destination_q_key == self._logging_message_q_key:
                            processed_sub_item.process_id = self._process_id
                            processed_sub_item.process_type = self._process_type
                        self._producers[destination_q_key].put_item(processed_sub_item, stop_event=self._stop_event)
