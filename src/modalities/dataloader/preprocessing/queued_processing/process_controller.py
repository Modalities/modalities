import multiprocessing as mp
from dataclasses import dataclass
from multiprocessing.synchronize import Event

import tqdm

from modalities.dataloader.preprocessing.queued_processing.processors import Processor
from modalities.utils.logging import get_logger


@dataclass
class PipelineStep:
    name: str
    poisonable: bool
    input_queue: mp.Queue
    processors: list[Processor]


class ProcessController:
    def __init__(self, pipeline_steps: list[PipelineStep], stop_event: Event, join_timeout: int = 5):
        """Initializes the ProcessController
        Each pipeline step contains a list of processors that retrieve the data from the input queue,
        process it and if necessary put it into the output queue of the next step.
        """
        self._pipeline_steps = pipeline_steps
        self._stop_event = stop_event
        self._join_timeout = join_timeout

    def join_processors_in_step(self, step: PipelineStep):
        """Joins the processors of a pipeline step
        If the stop_event is set, the processors are terminated
        """
        # poison the input queues of the processors
        if step.poisonable:
            for _ in tqdm.tqdm(step.processors, desc=f"Poisoning {step.name} processes"):
                if step.input_queue is not None:
                    step.input_queue.put(None)

        # join the processors
        num_exits = 0
        while num_exits < len(step.processors):
            processor = step.processors[num_exits]

            # if the processor is not alive, we continue with the next one
            if not processor.is_alive():
                get_logger().info(f"Processor {processor.full_name} is not alive. Continuing with the next processor.")
                num_exits += 1
                continue
            # if the stop event is set, we terminate the processor
            if self._stop_event.is_set():
                try:
                    processor.terminate()
                except Exception as e:
                    # if we can't terminate the processor, we continue with the next one
                    get_logger().error(
                        f"Error while terminating processor {processor.full_name}: {e}. "
                        "Continuing with the next processor."
                    )
                    num_exits += 1
                    continue
                get_logger().info(f"Terminated processor {processor.full_name}")
                num_exits += 1
            # if the stop event is not set, we join the processor
            else:
                get_logger().info(f"Joining {processor.full_name} ...")
                processor.join(timeout=self._join_timeout)
                if processor.exitcode is None:
                    get_logger().info(f"Joining {processor.full_name} timed out. Exit code: {processor.exitcode} ...")
                    continue
                get_logger().info(f"Joined processor {processor.full_name}. Exit code: {processor.exitcode}")
                num_exits += 1

    def run(self):
        # start the processors
        for step in self._pipeline_steps:
            get_logger().info(f"Starting processors for step {step.name}")
            for processor in step.processors:
                processor.start()

        # wait for the processors to finish
        for step in self._pipeline_steps:
            get_logger().info(f"Stopping {step.name} processes...")
            self.join_processors_in_step(step)
