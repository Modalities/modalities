"""Tests for auxiliary-loss-free MoE load balancing and the auxiliary-loss plumbing."""

import pytest
import torch
from torch.optim import AdamW

from modalities.batch import InferenceResultBatch
from modalities.loss_functions import CLMCrossEntropyLoss
from modalities.models.components.moe.load_balancing import (
    MoEBalancing,
    get_expert_load_reduction_group,
    get_moe_layers,
    update_expert_biases,
)
from modalities.models.components.moe.moe_losses import MoEAuxLoss, WeightedSumLoss
from modalities.models.nemotron.nemotron_model import NemotronLLM
from tests.models.nemotron.test_nemotron_model import VOCAB_SIZE, _layer_specs, _make_model


def _moe_model(aux_loss_coeff: float = 0.0, **overrides) -> NemotronLLM:
    return _make_model(
        layer_pattern="EE",
        n_layer=2,
        layer_specs=_layer_specs(aux_loss_coeff=aux_loss_coeff),
        **overrides,
    )


# --------------------------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------------------------


def test_get_moe_layers_finds_layers_in_a_mixed_stack():
    model = _make_model(layer_pattern="MEE*-", n_layer=5)
    assert len(get_moe_layers(model)) == 2


def test_get_moe_layers_returns_empty_for_a_dense_model():
    assert get_moe_layers(_make_model(layer_pattern="M-", n_layer=2)) == []


def test_reduction_group_is_none_without_distributed():
    # Single-process training needs no collective at all.
    assert get_expert_load_reduction_group(device_mesh=None) is None


# --------------------------------------------------------------------------------------------
# The bias update rule
# --------------------------------------------------------------------------------------------


def test_update_moves_bias_up_for_underloaded_and_down_for_overloaded_experts():
    model = _moe_model()
    moe = get_moe_layers(model)[0]
    # Expert 0 is starved, expert 7 is swamped, the rest sit at the mean.
    counts = torch.tensor([0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 30.0])
    with torch.no_grad():
        moe.router.tokens_per_expert.copy_(counts)

    update_rate = 1e-3
    update_expert_biases([moe], update_rate=update_rate, process_group=None)

    bias = moe.router.expert_bias
    assert bias[0].item() == pytest.approx(update_rate)
    assert bias[7].item() == pytest.approx(-update_rate)
    # Mean load is 11.25, so the experts at 10 are slightly under-loaded.
    for expert_idx in range(1, 7):
        assert bias[expert_idx].item() == pytest.approx(update_rate)


def test_update_uses_a_fixed_step_size_regardless_of_imbalance_magnitude():
    # The rule depends only on the sign of the deviation, which is what makes it robust to
    # activation-checkpointing double counting and to gradient accumulation.
    model = _moe_model()
    moe = get_moe_layers(model)[0]

    for scale in (1.0, 1000.0):
        with torch.no_grad():
            moe.router.expert_bias.zero_()
            moe.router.tokens_per_expert.copy_(torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) * scale)
        update_expert_biases([moe], update_rate=1e-3, process_group=None)
        assert moe.router.expert_bias[0].item() == pytest.approx(1e-3)


def test_update_resets_the_token_counters():
    model = _moe_model()
    moe = get_moe_layers(model)[0]
    with torch.no_grad():
        moe.router.tokens_per_expert.fill_(5.0)
    update_expert_biases([moe], update_rate=1e-3, process_group=None)
    torch.testing.assert_close(moe.router.tokens_per_expert, torch.zeros_like(moe.router.tokens_per_expert))


def test_update_is_a_noop_when_no_tokens_were_routed():
    model = _moe_model()
    moe = get_moe_layers(model)[0]
    with torch.no_grad():
        moe.router.expert_bias.fill_(0.5)
    update_expert_biases([moe], update_rate=1e-3, process_group=None)
    torch.testing.assert_close(moe.router.expert_bias, torch.full_like(moe.router.expert_bias, 0.5))


def test_update_skips_layers_without_an_expert_bias():
    model = _make_model(
        layer_pattern="E",
        n_layer=1,
        layer_specs={**_layer_specs(), "E": _layer_specs()["E"]},
    )
    moe = get_moe_layers(model)[0]
    moe.router.expert_bias = None
    with torch.no_grad():
        moe.router.tokens_per_expert.fill_(3.0)
    # Must not raise, and must leave the counter alone (nothing to balance).
    update_expert_biases([moe], update_rate=1e-3, process_group=None)


def test_repeated_updates_drive_the_load_towards_balance():
    # A small closed-loop simulation: the bias should progressively favour the starved expert.
    model = _moe_model()
    moe = get_moe_layers(model)[0]
    starved = 3
    for _ in range(50):
        counts = torch.full((8,), 10.0)
        counts[starved] = 0.0
        with torch.no_grad():
            moe.router.tokens_per_expert.copy_(counts)
        update_expert_biases([moe], update_rate=1e-2, process_group=None)

    bias = moe.router.expert_bias
    assert bias[starved] > bias.mean()
    assert bias[starved].item() == pytest.approx(50 * 1e-2, rel=1e-3)


def test_expert_bias_influences_routing_after_the_update():
    # The end-to-end effect: balancing must actually change which experts get picked.
    model = _moe_model().eval()
    moe = get_moe_layers(model)[0]
    x = torch.randn(64, model.n_embd)
    _, before, _ = moe.router(x)

    starved = int(torch.bincount(before.reshape(-1), minlength=8).argmin())
    with torch.no_grad():
        counts = torch.full((8,), 100.0)
        counts[starved] = 0.0
        moe.router.tokens_per_expert.copy_(counts)
    # A large step so that the effect is unambiguous.
    update_expert_biases([moe], update_rate=10.0, process_group=None)

    _, after, _ = moe.router(x)
    assert (after == starved).sum() > (before == starved).sum()


# --------------------------------------------------------------------------------------------
# The optimizer decorator
# --------------------------------------------------------------------------------------------


def test_decorator_returns_the_same_optimizer_instance():
    model = _moe_model()
    optimizer = AdamW(model.parameters(), lr=1e-4)
    decorated = MoEBalancing.register_expert_bias_update_hook(
        optimizer=optimizer, model=model, expert_bias_update_rate=1e-3
    )
    assert decorated is optimizer


def test_hook_fires_once_per_optimizer_step_not_per_micro_batch():
    # Correctness under gradient accumulation: the bias must move by exactly one step size per
    # optimizer step, no matter how many forward/backward passes happened in between.
    model = _moe_model()
    optimizer = AdamW(model.parameters(), lr=0.0)
    MoEBalancing.register_expert_bias_update_hook(optimizer=optimizer, model=model, expert_bias_update_rate=1e-3)
    moe = get_moe_layers(model)[0]

    gradient_accumulation_steps = 4
    for _ in range(gradient_accumulation_steps):
        out = model({"input_ids": torch.randint(0, VOCAB_SIZE, (2, 8))})
        out["logits"].float().mean().backward()
    # All four micro-batches have accumulated into the counter.
    assert moe.router.tokens_per_expert.sum() == 2 * 8 * 2 * gradient_accumulation_steps

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # Exactly one update of magnitude update_rate was applied, and the counters were reset.
    assert moe.router.expert_bias.abs().max().item() == pytest.approx(1e-3)
    assert moe.router.tokens_per_expert.sum() == 0


def test_hook_updates_every_moe_layer():
    model = _moe_model()
    optimizer = AdamW(model.parameters(), lr=0.0)
    MoEBalancing.register_expert_bias_update_hook(optimizer=optimizer, model=model, expert_bias_update_rate=1e-3)
    model({"input_ids": torch.randint(0, VOCAB_SIZE, (2, 8))})["logits"].float().mean().backward()
    optimizer.step()

    for moe in get_moe_layers(model):
        assert moe.router.expert_bias.abs().max() > 0


def test_decorator_warns_and_is_inert_without_moe_layers(caplog):
    # A pipeline stage may legitimately hold no MoE layers.
    model = _make_model(layer_pattern="M-", n_layer=2)
    optimizer = AdamW(model.parameters(), lr=0.0)
    with caplog.at_level("WARNING"):
        MoEBalancing.register_expert_bias_update_hook(optimizer=optimizer, model=model, expert_bias_update_rate=1e-3)
    assert "No MoE layers found" in caplog.text
    optimizer.step()  # must not raise


def test_decorator_warns_when_expert_bias_is_disabled(caplog):
    specs = _layer_specs()
    from modalities.models.components.moe.experts import ExpertsBackend
    from modalities.models.nemotron.nemotron_layer_specs import NemotronMoELayerSpec
    from tests.models.nemotron.test_nemotron_model import N_EMBD, NORM_CONFIG

    specs["E"] = NemotronMoELayerSpec(
        n_embd=N_EMBD,
        num_experts=8,
        moe_ffn_hidden=32,
        top_k=2,
        use_expert_bias=False,
        experts_backend=ExpertsBackend.LOOPED,
        norm_config=NORM_CONFIG,
    )
    model = _make_model(layer_pattern="E", n_layer=1, layer_specs=specs)
    optimizer = AdamW(model.parameters(), lr=0.0)
    with caplog.at_level("WARNING"):
        MoEBalancing.register_expert_bias_update_hook(optimizer=optimizer, model=model, expert_bias_update_rate=1e-3)
    assert "none of them maintains an expert bias" in caplog.text


def test_decorator_rejects_non_positive_update_rate():
    model = _moe_model()
    optimizer = AdamW(model.parameters(), lr=1e-4)
    with pytest.raises(ValueError, match="must be positive"):
        MoEBalancing.register_expert_bias_update_hook(optimizer=optimizer, model=model, expert_bias_update_rate=0.0)


def test_expert_bias_survives_a_state_dict_round_trip():
    # The bias is training state that must be checkpointed, or balancing restarts from scratch on
    # every warmstart.
    model = _moe_model()
    moe = get_moe_layers(model)[0]
    with torch.no_grad():
        moe.router.expert_bias.copy_(torch.linspace(-1.0, 1.0, 8))

    restored = _moe_model()
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(get_moe_layers(restored)[0].router.expert_bias, moe.router.expert_bias)


# --------------------------------------------------------------------------------------------
# Auxiliary loss plumbing
# --------------------------------------------------------------------------------------------


def _forward_batch(model: NemotronLLM, batch_size: int = 2, seq_len: int = 8) -> InferenceResultBatch:
    inputs = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))
    targets = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))
    return InferenceResultBatch(targets={"target_ids": targets}, predictions=model({"input_ids": inputs}))


def test_moe_aux_loss_reads_the_value_from_the_forward_batch():
    model = _moe_model(aux_loss_coeff=1e-2, aux_loss_key="moe_aux_loss")
    batch = _forward_batch(model)
    aux_loss = MoEAuxLoss(prediction_key="moe_aux_loss")
    torch.testing.assert_close(aux_loss(batch), model.get_aux_loss())


def test_inference_result_batch_length_is_unaffected_by_the_scalar_aux_loss():
    # InferenceResultBatch derives its length from the *first* predictions entry, so the logits
    # must be inserted before the scalar auxiliary loss.
    model = _moe_model(aux_loss_coeff=1e-2, aux_loss_key="moe_aux_loss")
    batch = _forward_batch(model, batch_size=3)
    assert len(batch) == 3


def test_weighted_sum_loss_combines_terms():
    model = _moe_model(aux_loss_coeff=1e-2, aux_loss_key="moe_aux_loss")
    batch = _forward_batch(model)

    clm = CLMCrossEntropyLoss(target_key="target_ids", prediction_key="logits")
    aux = MoEAuxLoss(prediction_key="moe_aux_loss")
    combined = WeightedSumLoss(losses=[clm, aux], weights=[1.0, 1.0])

    torch.testing.assert_close(combined(batch), clm(batch) + aux(batch), rtol=1e-5, atol=1e-7)


def test_weighted_sum_loss_applies_the_weights():
    model = _moe_model(aux_loss_coeff=1e-2, aux_loss_key="moe_aux_loss")
    batch = _forward_batch(model)
    clm = CLMCrossEntropyLoss(target_key="target_ids", prediction_key="logits")
    aux = MoEAuxLoss(prediction_key="moe_aux_loss")

    weighted = WeightedSumLoss(losses=[clm, aux], weights=[0.5, 3.0])
    torch.testing.assert_close(weighted(batch), 0.5 * clm(batch) + 3.0 * aux(batch), rtol=1e-5, atol=1e-7)


def test_weighted_sum_loss_is_differentiable_into_the_router():
    model = _moe_model(aux_loss_coeff=1e-2, aux_loss_key="moe_aux_loss")
    batch = _forward_batch(model)
    combined = WeightedSumLoss(
        losses=[
            CLMCrossEntropyLoss(target_key="target_ids", prediction_key="logits"),
            MoEAuxLoss(prediction_key="moe_aux_loss"),
        ],
        weights=[1.0, 1.0],
    )
    combined(batch).backward()
    for moe in get_moe_layers(model):
        assert moe.router.gate.weight.grad is not None
        assert moe.router.gate.weight.grad.abs().sum() > 0


def test_weighted_sum_loss_rejects_mismatched_weights():
    clm = CLMCrossEntropyLoss(target_key="target_ids", prediction_key="logits")
    with pytest.raises(ValueError, match="they must match"):
        WeightedSumLoss(losses=[clm], weights=[1.0, 2.0])
    with pytest.raises(ValueError, match="at least one loss"):
        WeightedSumLoss(losses=[], weights=[])


def test_moe_aux_loss_raises_a_clear_error_when_the_key_is_absent():
    # A likely misconfiguration: model.aux_loss_key not matching the loss's prediction_key.
    model = _moe_model(aux_loss_coeff=1e-2, aux_loss_key="moe_aux_loss")
    batch = _forward_batch(model)
    with pytest.raises(Exception, match="not present in predictions"):
        MoEAuxLoss(prediction_key="wrong_key")(batch)


def test_registry_exposes_the_phase_four_components():
    from modalities.registry.components import COMPONENTS
    from modalities.registry.registry import Registry

    registry = Registry(COMPONENTS)
    assert registry.get_component("loss", "moe_aux_loss") is MoEAuxLoss
    assert registry.get_component("loss", "weighted_sum") is WeightedSumLoss
    assert registry.get_component("optimizer", "moe_load_balanced") is not None
