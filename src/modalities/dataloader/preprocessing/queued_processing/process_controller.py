import multiprocessing as mp
from dataclasses import dataclass
from typing import Callable

import tqdm

from modalities.dataloader.preprocessing.queued_processing.queued_processing import Processor
from modalities.utils.logging import get_logger


@dataclass
class PipelineStep:
    name: str
    input_queue: mp.Queue
    processors: list[Processor]


class ProcessController:
    def __init__(self, pipeline_steps: list[PipelineStep], populate_jobs: Callable):
        """Initializes the ProcessController
        Each pipeline step contains a list of processors that retrieve the data from the input queue,
        process it and if necessary put it into the output queue of the next step.
        """
        self._pipeline_steps = pipeline_steps
        self._populate_jobs = populate_jobs

    def run(self):
        # add the jobs to the input queues
        get_logger().info("Populating jobs")
        self._populate_jobs()

        # start the processors
        for step in self._pipeline_steps:
            get_logger().info(f"Starting processors for step {step.name}")
            for processor in step.processors:
                processor.start()

        # wait for the processors to finish
        for step in self._pipeline_steps:
            get_logger().info(f"Stopping {step.name} processes...")
            for _ in tqdm.tqdm(step.processors, desc=f"Poisoning {step.name} processes"):
                step.input_queue.put(None)
            get_logger().info(f"Waiting for processors in step {step.name} to finish")

            for processor in tqdm.tqdm(step.processors, desc=f"Joining {step.name} processes"):
                processor.join()
