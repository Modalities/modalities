from unittest.mock import MagicMock

import pytest
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

from modalities.checkpointing.stateful import app_state as app_state_module
from modalities.checkpointing.stateful.app_state import AppState, StatefulComponents


@pytest.fixture
def model() -> nn.Module:
    return nn.Linear(4, 2)


@pytest.fixture
def optimizer(model: nn.Module) -> SGD:
    return SGD(model.parameters(), lr=0.1)


@pytest.fixture
def lr_scheduler(optimizer: SGD) -> StepLR:
    return StepLR(optimizer, step_size=1)


@pytest.fixture
def patched_retrievers(monkeypatch: pytest.MonkeyPatch) -> dict[StatefulComponents, MagicMock]:
    """Replace each retriever's ``load_state_dict_`` with a mock so we can assert which ones were invoked."""
    mocks = {
        StatefulComponents.MODEL: MagicMock(),
        StatefulComponents.OPTIMIZER: MagicMock(),
        StatefulComponents.LR_SCHEDULER: MagicMock(),
    }
    monkeypatch.setattr(app_state_module.ModelStateRetriever, "load_state_dict_", mocks[StatefulComponents.MODEL])
    monkeypatch.setattr(
        app_state_module.OptimizerStateRetriever, "load_state_dict_", mocks[StatefulComponents.OPTIMIZER]
    )
    monkeypatch.setattr(
        app_state_module.LRSchedulerStateRetriever, "load_state_dict_", mocks[StatefulComponents.LR_SCHEDULER]
    )
    return mocks


def _make_state_dict() -> dict:
    return {
        StatefulComponents.MODEL.value: {"model_payload": True},
        StatefulComponents.OPTIMIZER.value: {"optimizer_payload": True},
        StatefulComponents.LR_SCHEDULER.value: {"lr_scheduler_payload": True},
    }


class TestComponentsToLoad:
    def test_default_without_lr_scheduler_loads_model_and_optimizer(
        self, model: nn.Module, optimizer: SGD, patched_retrievers: dict[StatefulComponents, MagicMock]
    ) -> None:
        app_state = AppState(model=model, optimizer=optimizer)

        app_state.load_state_dict(_make_state_dict())

        patched_retrievers[StatefulComponents.MODEL].assert_called_once()
        patched_retrievers[StatefulComponents.OPTIMIZER].assert_called_once()
        patched_retrievers[StatefulComponents.LR_SCHEDULER].assert_not_called()
        assert app_state.is_loaded

    def test_default_with_lr_scheduler_loads_all_three(
        self,
        model: nn.Module,
        optimizer: SGD,
        lr_scheduler: StepLR,
        patched_retrievers: dict[StatefulComponents, MagicMock],
    ) -> None:
        app_state = AppState(model=model, optimizer=optimizer, lr_scheduler=lr_scheduler)

        app_state.load_state_dict(_make_state_dict())

        patched_retrievers[StatefulComponents.MODEL].assert_called_once()
        patched_retrievers[StatefulComponents.OPTIMIZER].assert_called_once()
        patched_retrievers[StatefulComponents.LR_SCHEDULER].assert_called_once()

    @pytest.mark.parametrize(
        "selected",
        [
            [StatefulComponents.MODEL],
            [StatefulComponents.OPTIMIZER],
            [StatefulComponents.LR_SCHEDULER],
            [StatefulComponents.MODEL, StatefulComponents.OPTIMIZER],
            [StatefulComponents.MODEL, StatefulComponents.LR_SCHEDULER],
            [],
        ],
    )
    def test_explicit_selection_only_loads_chosen_components(
        self,
        model: nn.Module,
        optimizer: SGD,
        lr_scheduler: StepLR,
        patched_retrievers: dict[StatefulComponents, MagicMock],
        selected: list[StatefulComponents],
    ) -> None:
        app_state = AppState(model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, components_to_load=selected)

        app_state.load_state_dict(_make_state_dict())

        for component, mock in patched_retrievers.items():
            if component in selected:
                mock.assert_called_once()
            else:
                mock.assert_not_called()

    def test_lr_scheduler_in_components_but_no_scheduler_attached_is_skipped(
        self, model: nn.Module, optimizer: SGD, patched_retrievers: dict[StatefulComponents, MagicMock]
    ) -> None:
        # Guards against the lr_scheduler branch firing when no scheduler is attached — the
        # state_dict won't carry a scheduler entry, so the retriever must not be called.
        app_state = AppState(
            model=model,
            optimizer=optimizer,
            components_to_load=[StatefulComponents.MODEL, StatefulComponents.LR_SCHEDULER],
        )

        state_dict = _make_state_dict()
        state_dict.pop(StatefulComponents.LR_SCHEDULER.value)

        app_state.load_state_dict(state_dict)

        patched_retrievers[StatefulComponents.MODEL].assert_called_once()
        patched_retrievers[StatefulComponents.OPTIMIZER].assert_not_called()
        patched_retrievers[StatefulComponents.LR_SCHEDULER].assert_not_called()

    def test_double_load_raises(
        self, model: nn.Module, optimizer: SGD, patched_retrievers: dict[StatefulComponents, MagicMock]
    ) -> None:
        app_state = AppState(model=model, optimizer=optimizer)
        app_state.load_state_dict(_make_state_dict())

        with pytest.raises(RuntimeError):
            app_state.load_state_dict(_make_state_dict())
