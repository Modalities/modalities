import json
import tempfile
from pathlib import Path

import torch

from modalities.batch import EvaluationResultBatch, ResultItem
from modalities.logging_broker.messages import Message, MessageTypes
from modalities.logging_broker.subscriber_impl.results_subscriber import EvaluationResultToDiscSubscriber


class DummyEvalResult(EvaluationResultBatch):
    def __init__(self):
        # Minimal dummy values for required fields
        self.dataloader_tag = "test"
        self.losses = {"loss": ResultItem(torch.tensor(1.23))}
        self.metrics = {"acc": ResultItem(torch.tensor(0.99))}
        self.throughput_metrics = {"throughput": ResultItem(torch.tensor(100.0))}
        self.num_train_steps_done = 1
        self.train_local_sample_id = 0


def test_consume_message_writes_jsonl(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "results.jsonl"
        subscriber = EvaluationResultToDiscSubscriber(out_path)
        num_records = 5
        for i in range(num_records):

            class DummyEvalResult(EvaluationResultBatch):
                def __init__(self):
                    self.dataloader_tag = f"tag_{i}"
                    self.losses = {"loss": ResultItem(torch.tensor(float(i)))}
                    self.metrics = {"acc": ResultItem(torch.tensor(float(i + 1)))}
                    self.throughput_metrics = {"throughput": ResultItem(torch.tensor(float(i + 2)))}
                    self.num_train_steps_done = i
                    self.train_local_sample_id = i

            eval_result = DummyEvalResult()
            msg = Message(message_type=MessageTypes.EVALUATION_RESULT, payload=eval_result)
            subscriber.consume_message(msg)
        # Check file exists and content is valid JSON
        with out_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == num_records
            for idx, line in enumerate(lines):
                data = json.loads(line)
                assert data["dataloader_tag"] == f"tag_{idx}"
                assert abs(data["losses"]["loss"] - float(idx)) < 1e-6
                assert abs(data["metrics"]["acc"] - float(idx + 1)) < 1e-6
                assert abs(data["throughput_metrics"]["throughput"] - float(idx + 2)) < 1e-6
                assert data["num_train_steps_done"] == idx
