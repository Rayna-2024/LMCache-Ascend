# SPDX-License-Identifier: Apache-2.0
"""CPU numerical checks against independent HuggingFace Qwen3 full prefill."""

# Standard
from types import SimpleNamespace

# Third Party
from transformers import Qwen3Config, Qwen3Model
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
import pytest
import torch
import torch.nn.functional as F

# First Party
from lmcache_ascend.v1.blend.mp_selective import (
    SelectiveConfig,
    select_positions,
    selective_prefill,
)


def attention(q, k, v, positions, scale):
    allowed = torch.arange(k.shape[0])[None, :] <= positions[:, None]
    return F.scaled_dot_product_attention(
        q.transpose(0, 1)[None],
        k.transpose(0, 1)[None],
        v.transpose(0, 1)[None],
        attn_mask=allowed,
        scale=scale,
        enable_gqa=True,
    )[0].transpose(0, 1)


def cpu_transfer(buffer, caches, slots, from_paged):
    """CPU reference for the public fused transfer contract."""
    valid = slots >= 0
    for view, cache in zip(buffer, caches, strict=True):
        paged = cache.flatten(0, 1).flatten(1)
        if from_paged:
            view[valid] = paged[slots[valid]]
        else:
            paged[slots[valid]] = view[valid]


def norm_adapter(norm):
    def call(x, residual=None):
        if residual is None:
            return norm(x)
        total = x + residual
        return norm(total), total

    return call


def model_adapter(model):
    """Adapt public HF weights to vLLM's fused projection/norm interface."""
    layers = []
    for index, layer in enumerate(model.layers):
        a = layer.self_attn

        def project(x, a=a):
            return torch.cat((a.q_proj(x), a.k_proj(x), a.v_proj(x)), dim=-1), None

        def rope(positions, q, k):
            cos, sin = model.rotary_emb(q[None], positions[None])
            qh = q.reshape(1, -1, 4, 8).transpose(1, 2)
            kh = k.reshape(1, -1, 2, 8).transpose(1, 2)
            qh, kh = apply_rotary_pos_emb(qh, kh, cos, sin)
            return qh.transpose(1, 2).reshape(q.shape), kh.transpose(1, 2).reshape(
                k.shape
            )

        layers.append(
            SimpleNamespace(
                input_layernorm=norm_adapter(layer.input_layernorm),
                post_attention_layernorm=norm_adapter(layer.post_attention_layernorm),
                mlp=layer.mlp,
                self_attn=SimpleNamespace(
                    qkv_proj=project,
                    q_norm=a.q_norm,
                    k_norm=a.k_norm,
                    rotary_emb=rope,
                    q_size=32,
                    kv_size=16,
                    num_heads=4,
                    num_kv_heads=2,
                    head_dim=8,
                    scaling=8**-0.5,
                    o_proj=lambda x, a=a: (a.o_proj(x), None),
                    attn=SimpleNamespace(
                        layer_name=str(index),
                        impl=SimpleNamespace(use_bnsd_kv_cache=False),
                    ),
                ),
            )
        )
    return SimpleNamespace(
        model=SimpleNamespace(layers=layers, norm=norm_adapter(model.norm)),
        embed_input_ids=model.embed_tokens,
        reference=model,
    )


def setup_model():
    torch.manual_seed(354)
    cfg = Qwen3Config(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        attention_dropout=0.0,
    )
    cfg._attn_implementation = "eager"
    model = Qwen3Model(cfg).eval()
    old = [1, 2, 3, 10, 11, 12, 13, 14]
    new = [4, 5, 6, 10, 11, 12, 13, 14]
    with torch.no_grad():
        old_kv = model(torch.tensor([old]), use_cache=True).past_key_values
        fresh = model(torch.tensor([new]), use_cache=True).past_key_values
    # Non-monotonic physical block IDs catch logical/physical indexing errors.
    table = [2, 0, 3, 1]
    slots = torch.tensor(table).repeat_interleave(2) * 2 + torch.arange(8) % 2
    caches = {}
    for i in range(3):
        pair = [torch.full((5, 2, 2, 8), float("nan")) for _ in range(2)]
        for dst, value in zip(
            pair, (old_kv.layers[i].keys, old_kv.layers[i].values), strict=True
        ):
            dst.view(10, 2, 8)[slots[3:]] = value[0].transpose(0, 1)[3:]
        caches[str(i)] = pair
    return model_adapter(model), caches, new, table, slots, fresh


@pytest.mark.parametrize("hits", [[(3, 8)], []])
def test_ratio_one_matches_hf_full_recompute_and_preserves_other_blocks(hits):
    model, caches, tokens, table, slots, fresh = setup_model()
    stats = selective_prefill(
        model,
        caches,
        tokens,
        table,
        hits,
        SelectiveConfig(1.0),
        attention,
        transfer=cpu_transfer,
    )
    assert stats["layer_query_counts"] == [8, 8, 8]
    for i in range(3):
        for actual, expected in zip(
            caches[str(i)], (fresh.layers[i].keys, fresh.layers[i].values), strict=True
        ):
            torch.testing.assert_close(
                actual.view(10, 2, 8)[slots],
                expected[0].transpose(0, 1),
                atol=1e-6,
                rtol=1e-5,
            )
            assert actual[4].isnan().all()


@pytest.mark.parametrize("ratio", [0.15, 1.0])
def test_merged_tail_matches_dense_reference(ratio):
    model, caches, tokens, table, slots, fresh = setup_model()
    # Seed hits with the dense reference so their reuse has an exact oracle.
    for i in range(3):
        for dst, src in zip(caches[str(i)], (fresh.layers[i].keys, fresh.layers[i].values), strict=True):
            dst.view(10, 2, 8)[slots[3:6]] = src[0].transpose(0, 1)[3:6]
    stats = selective_prefill(
        model, caches, tokens, table, [(3, 6)], SelectiveConfig(ratio, debug_selected_positions=True),
        attention, transfer=cpu_transfer, output_tail_tokens=2,
    )
    assert stats["selected_positions"][-2:] == [6, 7]
    with torch.no_grad():
        expected = model.reference(torch.tensor([tokens])).last_hidden_state[0, -2:]
    torch.testing.assert_close(stats["tail_hidden_states"], expected, atol=1e-6, rtol=1e-5)
    for i in range(3):
        for dst, src in zip(caches[str(i)], (fresh.layers[i].keys, fresh.layers[i].values), strict=True):
            torch.testing.assert_close(
                dst.view(10, 2, 8)[slots[-2:]], src[0].transpose(0, 1)[-2:],
                atol=1e-6, rtol=1e-5,
            )


def test_sparse_prefill_keeps_misses_and_preserves_unselected_cached_kv():
    model, caches, tokens, table, slots, _ = setup_model()
    before = {name: [x.clone() for x in pair] for name, pair in caches.items()}
    stats = selective_prefill(
        model,
        caches,
        tokens,
        table,
        [(3, 8)],
        SelectiveConfig(0.2, debug_selected_positions=True),
        attention,
        transfer=cpu_transfer,
    )
    selected = stats["selected_positions"]
    assert set(range(3)).issubset(selected)
    assert len(selected) == 4
    assert stats["layer_query_counts"] == [8, 4, 4]
    remaining = [i for i in range(8) if i not in selected]
    for i in [1, 2]:
        for actual, old in zip(caches[str(i)], before[str(i)], strict=True):
            assert torch.isfinite(actual.view(10, 2, 8)[slots]).all()
            torch.testing.assert_close(
                actual.view(10, 2, 8)[slots[remaining]],
                old.view(10, 2, 8)[slots[remaining]],
                rtol=0,
                atol=0,
            )


@pytest.mark.parametrize(
    "ratio,expected_hits",
    [(0, []), (0.15, [5]), (0.5, [5, 8, 10]), (1.0, [1, 3, 4, 5, 7, 8, 10])],
)
def test_selection_chooses_largest_deviations_with_ratio_budget(ratio, expected_hits):
    fresh = torch.zeros(11, 2)
    cached = torch.full_like(fresh, float("nan"))
    hits = torch.tensor([1, 3, 4, 5, 7, 8, 10])
    cached[hits] = torch.tensor(
        [
            [1.0, 0.0],
            [4.0, 0.0],
            [2.0, 0.0],
            [7.0, 0.0],
            [3.0, 0.0],
            [6.0, 0.0],
            [5.0, 0.0],
        ]
    )
    # Squared errors rank hits as 5, 8, 10, 3, 7, 4, 1.
    # Budgets for seven hits are 0, 1, 3, and 7; gaps always require computation.
    def unexpected_collective(_):
        pytest.fail("0% selection must not call a scoring collective")

    kwargs = {"reduce_scores": unexpected_collective, "broadcast": unexpected_collective} if ratio == 0 else {}
    selected = select_positions(fresh, cached, hits, ratio, **kwargs)
    assert selected.tolist() == sorted([0, 2, 6, 9] + expected_hits)


@pytest.mark.parametrize("ratio,hits", [(1.0, [(3, 8)]), (0.2, [(3, 8)]), (0.2, [])])
def test_ready_prefix_is_read_but_never_computed_or_overwritten(ratio, hits):
    model, caches, tokens, table, slots, fresh = setup_model()
    prefix = 2
    for i in range(3):
        for cache, expected in zip(
            caches[str(i)], (fresh.layers[i].keys, fresh.layers[i].values), strict=True
        ):
            cache.view(10, 2, 8)[slots[:prefix]] = expected[0].transpose(0, 1)[:prefix]
    before = {name: [x.clone() for x in pair] for name, pair in caches.items()}
    stats = selective_prefill(
        model,
        caches,
        tokens,
        table,
        hits,
        SelectiveConfig(ratio, debug_selected_positions=True),
        attention,
        transfer=cpu_transfer,
        computed_prefix_tokens=prefix,
    )
    count = 6 if ratio == 1.0 or not hits else 2
    assert stats["layer_qkv_counts"] == [6, 6, count]
    assert stats["layer_query_counts"] == [6, count, count]
    assert stats["retained_prefix_tokens"] == prefix
    assert stats["mandatory_tokens"] == (1 if hits else 6)
    assert min(stats["selected_positions"]) >= prefix
    for i in range(3):
        for cache, old, expected in zip(
            caches[str(i)],
            before[str(i)],
            (fresh.layers[i].keys, fresh.layers[i].values),
            strict=True,
        ):
            torch.testing.assert_close(
                cache.view(10, 2, 8)[slots[:prefix]],
                old.view(10, 2, 8)[slots[:prefix]],
                rtol=0,
                atol=0,
            )
            assert cache[4].isnan().all()
            if ratio == 1.0 or not hits:
                torch.testing.assert_close(
                    cache.view(10, 2, 8)[slots],
                    expected[0].transpose(0, 1),
                    rtol=1e-5,
                    atol=1e-6,
                )


@pytest.mark.parametrize("loaded,success_count", [(False, 0), (True, 1)])
@pytest.mark.parametrize("tail", [0, 2])
def test_tp_load_failure_computes_coverage_before_tail(
    monkeypatch, loaded, success_count, tail
):
    """One failed rank forces even successful ranks to recompute all coverage."""
    # Standard
    from functools import partial

    # Third Party
    from lmcache.integration.vllm.mp_blend_client import BlendCoverageTask
    from lmcache.v1.multiprocess.custom_types import CBMatchResult, IPCCacheServerKey

    # First Party
    from lmcache_ascend.integration.vllm import mp_selective_blend as module

    model, caches, tokens, table, slots, fresh = setup_model()
    connector = object.__new__(module.MPSelectiveBlendConnector)
    connector.mla_reference = False
    connector.selective_caches = caches
    connector.selective_config = SelectiveConfig(0.2)
    connector.worker_adapter = SimpleNamespace(worker_id=0)
    group = SimpleNamespace(
        world_size=2,
        all_reduce=lambda value: torch.tensor([success_count]),
        broadcast=lambda value: value,
    )
    monkeypatch.setattr(module, "get_tp_group", lambda: group)
    monkeypatch.setattr(
        module,
        "selective_prefill",
        partial(
            selective_prefill,
            attention=attention,
            transfer=cpu_transfer,
        ),
    )
    key = IPCCacheServerKey.from_token_ids("qwen3", 2, None, tokens, request_id="r")
    coverage = len(tokens) - tail
    args = (key, [CBMatchResult(3, coverage, 3, coverage, b"hit")], [table], 0, 2, coverage)
    task = module.MergedCoverageTask(*args, tail_tokens=tail) if tail else BlendCoverageTask(*args)
    context = SimpleNamespace(model_instance=model)
    connector.complete_blend(task, loaded, context)
    assert not hasattr(context, "lmcache_merged_tail_hidden_states")
    for i in range(3):
        for actual, expected in zip(
            caches[str(i)], (fresh.layers[i].keys, fresh.layers[i].values), strict=True
        ):
            torch.testing.assert_close(
                actual.view(10, 2, 8)[slots[:coverage]],
                expected[0].transpose(0, 1)[:coverage],
                atol=1e-6,
                rtol=1e-5,
            )


@pytest.mark.parametrize("ratio", [0.0, 0.2, 1.0])
def test_debug_positions_does_not_change_kv_or_tail(monkeypatch, ratio):
    model, caches, tokens, table, _, _ = setup_model()
    debug_caches = {name: [x.clone() for x in pair] for name, pair in caches.items()}

    def unexpected_cpu_copy(*args, **kwargs):
        pytest.fail("Default selective stats must not copy positions to CPU")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", unexpected_cpu_copy)
        stats = selective_prefill(
            model, caches, tokens, table, [(3, 6)], SelectiveConfig(ratio),
            attention, transfer=cpu_transfer, output_tail_tokens=2,
        )
    debug = selective_prefill(
        model, debug_caches, tokens, table, [(3, 6)],
        SelectiveConfig(ratio, debug_selected_positions=True),
        attention, transfer=cpu_transfer, output_tail_tokens=2,
    )
    assert "selected_positions" not in stats
    assert len(debug["selected_positions"]) == stats["mandatory_tokens"] + stats["selected_hit_tokens"]
    assert stats.keys() == debug.keys() - {"selected_positions"}
    for key, value in stats.items():
        if key == "tail_hidden_states":
            torch.testing.assert_close(value, debug[key], rtol=0, atol=0)
        else:
            assert value == debug[key]
    for name, pair in caches.items():
        for actual, expected in zip(pair, debug_caches[name], strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("value", ["true", 1, None])
def test_debug_positions_requires_bool(value):
    with pytest.raises(ValueError, match="debug_selected_positions must be a bool"):
        SelectiveConfig(debug_selected_positions=value)
