from llm_gym.batch import EvaluationResultBatch
import torch.distributed as dist


class ResultsCallbackIF:
    def __call__(self, evaluation_result: EvaluationResultBatch):
        raise NotImplementedError


class DummyResultsCallback(ResultsCallbackIF):
    def __call__(self, evaluation_result: EvaluationResultBatch):
        pass


class ResultsCallback(ResultsCallbackIF):
    def __init__(self, subscribing_global_rank: int = None) -> None:
        self.subscribing_global_rank = subscribing_global_rank
        if dist.get_rank() == self.subscribing_global_rank:
            pass

    def __call__(self, evaluation_result: EvaluationResultBatch):
        if self.subscribing_global_rank is not None and dist.get_rank() == self.subscribing_global_rank:
            print(evaluation_result)
