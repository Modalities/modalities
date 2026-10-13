import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from pydantic import BaseModel

from modalities.config.component_factory import ComponentFactory
from modalities.evaluator import DownstreamEvaluator
from modalities.registry.components import COMPONENTS
from modalities.registry.registry import Registry
from modalities.tokenization.tokenizer_wrapper import TokenizerWrapper


# ---------- helpers ----------

class MockTokenizer(TokenizerWrapper):
    def tokenize(self, text: str) -> list[int]:
        return []

    def decode(self, input_ids: list[int]) -> str:
        return ""

    @property
    def vocab_size(self) -> int:
        return 0

    def get_token_id(self, token: str) -> int:
        return 0

    def is_special_token_id(self, token_id: int) -> bool:
        return False


# ---------- DownstreamEvaluator tests ----------

def test_downstream_evaluator_skips_non_matching_step():
    evaluator = DownstreamEvaluator(
        tokenizer=MockTokenizer(),
        tasks=["arc_challenge::olmes"],
        eval_interval=5,
        checkpoint_dir=Path("/tmp/fake"),
        global_rank=0,
        olmes_command_template="echo {hf_model_dir} {tasks} {step}",
    )
    with patch("subprocess.Popen") as mock_popen:
        evaluator.evaluate(num_train_steps_done=3)
        mock_popen.assert_not_called()


def test_downstream_evaluator_skips_non_rank_zero():
    evaluator = DownstreamEvaluator(
        tokenizer=MockTokenizer(),
        tasks=["arc_challenge::olmes"],
        eval_interval=5,
        checkpoint_dir=Path("/tmp/fake"),
        global_rank=1,
        olmes_command_template="echo {hf_model_dir} {tasks} {step}",
    )
    with patch("subprocess.Popen") as mock_popen:
        evaluator.evaluate(num_train_steps_done=5)
        mock_popen.assert_not_called()


def test_downstream_evaluator_runs_when_hf_checkpoint_exists():
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir)
        ckpt_path = checkpoint_dir / "step_10"
        ckpt_path.mkdir()
        hf_dir = ckpt_path / "hf_checkpoint"
        hf_dir.mkdir()

        info = {"checkpoint_folder_path": str(ckpt_path)}
        with open(checkpoint_dir / "last_checkpoint_info.json", "w") as f:
            json.dump(info, f)

        evaluator = DownstreamEvaluator(
            tokenizer=MockTokenizer(),
            tasks=["arc_challenge::olmes", "hellaswag::olmes"],
            eval_interval=10,
            checkpoint_dir=checkpoint_dir,
            global_rank=0,
            olmes_command_template="olmes --model {hf_model_dir} --tasks {tasks} --step {step}",
        )

        with patch("subprocess.Popen") as mock_popen:
            evaluator.evaluate(num_train_steps_done=10)
            mock_popen.assert_called_once()
            cmd = mock_popen.call_args[0][0]
            assert str(hf_dir) in cmd
            assert "arc_challenge::olmes hellaswag::olmes" in cmd
            assert "10" in cmd


def test_downstream_evaluator_skips_when_no_hf_checkpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir)
        ckpt_path = checkpoint_dir / "step_10"
        ckpt_path.mkdir()
        # No hf_checkpoint folder

        info = {"checkpoint_folder_path": str(ckpt_path)}
        with open(checkpoint_dir / "last_checkpoint_info.json", "w") as f:
            json.dump(info, f)

        evaluator = DownstreamEvaluator(
            tokenizer=MockTokenizer(),
            tasks=["arc_challenge::olmes"],
            eval_interval=10,
            checkpoint_dir=checkpoint_dir,
            global_rank=0,
            olmes_command_template="echo {hf_model_dir} {tasks} {step}",
        )

        with patch("subprocess.Popen") as mock_popen:
            evaluator.evaluate(num_train_steps_done=10)
            mock_popen.assert_not_called()


# ---------- Factory instantiation tests ----------

def test_downstream_evaluator_factory_instantiation():
    from modalities.config.pydantic_if_types import PydanticDownstreamEvaluatorType

    registry = Registry(COMPONENTS)
    component_factory = ComponentFactory(registry=registry)

    tokenizer_mock = MockTokenizer()

    class TrainingModel(BaseModel):
        downstream_eval: PydanticDownstreamEvaluatorType

    config_dict = {
        "downstream_eval": {
            "component_key": "downstream_evaluator",
            "variant_key": "default",
            "config": {
                "tokenizer": tokenizer_mock,
                "tasks": ["task_a"],
                "eval_interval": 10,
                "checkpoint_dir": "/tmp/test_checkpoints",
                "global_rank": 0,
                "olmes_command_template": "echo {hf_model_dir}",
            },
        }
    }

    components = component_factory.build_components(
        config_dict=config_dict,
        components_model_type=TrainingModel,
    )

    assert isinstance(components.downstream_eval, DownstreamEvaluator)
    assert components.downstream_eval.tokenizer == tokenizer_mock
    assert components.downstream_eval.tasks == ["task_a"]
    assert components.downstream_eval.eval_interval == 10
