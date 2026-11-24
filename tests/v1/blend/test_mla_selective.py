# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the experimental MLA accuracy path."""

from types import SimpleNamespace as NS

import pytest
import torch
import torch.nn.functional as F

from lmcache_ascend.v1.blend.mla_selective import (
    mla_selective_prefill,
    reference_attention,
    rotate_positions,
)
from lmcache_ascend.v1.blend.mp_selective import SelectiveConfig


class Linear(torch.nn.Linear):
    def forward(self, x, **kwargs):
        return super().forward(x), None


class Norm(torch.nn.Module):
    def forward(self, x, residual=None):
        total = x if residual is None else x + residual
        y = total * torch.rsqrt(total.square().mean(-1, keepdim=True) + 1e-6)
        return y if residual is None else (y, total)


def fixture():
    torch.manual_seed(61)
    embedding = torch.nn.Embedding(32, 12)
    angles = torch.arange(128)[:, None] * torch.tensor([[0.17, 0.031]])
    rope = NS(
        head_size=4,
        is_neox_style=False,
        cos_sin_cache=torch.cat((angles.cos(), angles.sin()), -1) * 1.1,
    )
    layers = []
    caches = {}
    for i in range(3):
        attn = NS(
            q_lora_rank=6,
            kv_lora_rank=5,
            qk_rope_head_dim=4,
            qk_nope_head_dim=4,
            qk_head_dim=8,
            v_head_dim=3,
            num_local_heads=2,
            fused_qkv_a_proj=Linear(12, 15, bias=False),
            q_a_layernorm=Norm(),
            kv_a_layernorm=Norm(),
            q_b_proj=Linear(6, 16, bias=False),
            kv_b_proj=Linear(5, 14, bias=False),
            o_proj=Linear(6, 12, bias=False),
            rotary_emb=rope,
            scaling=8**-0.5,
            mla_attn=NS(mla_attn=NS(layer_name=str(i))),
        )
        layers.append(
            NS(
                self_attn=attn,
                input_layernorm=Norm(),
                post_attention_layernorm=Norm(),
                mlp=torch.nn.Sequential(
                    torch.nn.Linear(12, 24), torch.nn.SiLU(), torch.nn.Linear(24, 12)
                ),
                use_sequence_parallel_moe=False,
            )
        )
        caches[str(i)] = (
            torch.full((4, 4, 1, 5), torch.nan),
            torch.full((4, 4, 1, 4), torch.nan),
        )
    return NS(model=NS(layers=layers, norm=Norm()), embed_input_ids=embedding), caches


def independent_full(model, ids, return_hidden=False):
    """Dense MLA oracle using explicit projection weights, complex RoPE and SDPA."""
    x = model.embed_input_ids(torch.tensor(ids))
    result = []
    positions = torch.arange(len(ids))
    for layer in model.model.layers:
        a = layer.self_attn
        residual = x
        h = layer.input_layernorm(x)
        low = F.linear(h, a.fused_qkv_a_proj.weight)
        qlow, c, kr = low.split((6, 5, 4), -1)
        q = F.linear(a.q_a_layernorm(qlow), a.q_b_proj.weight).reshape(-1, 2, 8)
        c = a.kv_a_layernorm(c)
        table = a.rotary_emb.cos_sin_cache[positions]
        phase = torch.complex(table[:, :2], table[:, 2:])[:, None]

        def rotate(z, phase=phase):
            zc = torch.view_as_complex(
                z.float().reshape(*z.shape[:-1], 2, 2).contiguous()
            )
            return torch.view_as_real(zc * phase).flatten(-2)

        qr = rotate(q[..., 4:])
        kr = rotate(kr[:, None])[:, 0]
        q = torch.cat((q[..., :4], qr), -1)
        k, v = F.linear(c, a.kv_b_proj.weight).reshape(-1, 2, 7).split((4, 3), -1)
        k = torch.cat((k, kr[:, None].expand(-1, 2, -1)), -1)
        y = F.scaled_dot_product_attention(
            q.transpose(0, 1),
            k.transpose(0, 1),
            v.transpose(0, 1),
            is_causal=True,
            scale=a.scaling,
        ).transpose(0, 1)
        x = residual + F.linear(y.flatten(1), a.o_proj.weight)
        x = x + layer.mlp(layer.post_attention_layernorm(x))
        result.append((c, kr))
    return (result, model.model.norm(x)) if return_hidden else result


def test_full_recompute_matches_independent_dense_mla():
    model, caches = fixture()
    ids = [1, 3, 5, 7, 9, 11, 13]
    expected = independent_full(model, ids)
    stats = mla_selective_prefill(
        model, caches, ids, [2, 0], [], SelectiveConfig(ratio=1)
    )
    slots = torch.tensor([8, 9, 10, 11, 0, 1, 2])
    for i, (c, r) in enumerate(expected):
        torch.testing.assert_close(
            caches[str(i)][0].flatten(0, 2)[slots], c, atol=2e-6, rtol=2e-5
        )
        torch.testing.assert_close(
            caches[str(i)][1].flatten(0, 2)[slots], r, atol=2e-6, rtol=2e-5
        )
    assert stats["layer_query_counts"] == [7, 7, 7]


def test_ratio_one_overwrites_stale_hits_and_preserves_ready_prefix():
    model, caches = fixture()
    ids = [1, 3, 5, 7, 9, 11, 13]
    expected = independent_full(model, ids)
    for i, (c, r) in enumerate(expected):
        caches[str(i)][0].flatten(0, 2)[:2] = c[:2]
        caches[str(i)][1].flatten(0, 2)[:2] = r[:2]
        caches[str(i)][0].flatten(0, 2)[2:6] = 123
        caches[str(i)][1].flatten(0, 2)[2:6] = -123
    stats = mla_selective_prefill(
        model,
        caches,
        ids,
        [0, 1],
        [(2, 6)],
        SelectiveConfig(ratio=1, debug_selected_positions=True),
        computed_prefix_tokens=2,
    )
    assert stats["selected_positions"] == [2, 3, 4, 5, 6]
    for i, (c, r) in enumerate(expected):
        torch.testing.assert_close(
            caches[str(i)][0].flatten(0, 2)[:7], c, atol=3e-6, rtol=2e-5
        )
        torch.testing.assert_close(
            caches[str(i)][1].flatten(0, 2)[:7], r, atol=3e-6, rtol=2e-5
        )


def test_partial_recompute_keeps_hits_and_fills_nan_misses():
    model, caches = fixture()
    ids = [1, 3, 5, 7, 9, 11, 13]
    old = independent_full(model, [2, 4, 6, 8, 10, 12, 14])
    for i, (c, r) in enumerate(old):
        caches[str(i)][0].flatten(0, 2)[1:6] = c[1:6]
        caches[str(i)][1].flatten(0, 2)[1:6] = r[1:6]
    stats = mla_selective_prefill(
        model, caches, ids, [0, 1], [(1, 6)], SelectiveConfig(ratio=0.4, debug_selected_positions=True)
    )
    selected = stats["selected_positions"]
    assert 0 in selected and 6 in selected and len(selected) == 4
    kept = [i for i in range(1, 6) if i not in selected]
    for layer in (1, 2):
        for plane in (0, 1):
            got = caches[str(layer)][plane].flatten(0, 2)
            assert torch.isfinite(got[:7]).all()
            torch.testing.assert_close(
                got[kept], old[layer][plane][kept], rtol=0, atol=0
            )


@pytest.mark.parametrize("ratio", [0, 0.4, 1.0])
def test_merged_tail_remains_mandatory_and_returns_normalized_hidden(ratio):
    model, caches = fixture()
    ids = [1, 3, 5, 7, 9, 11, 13]
    expected_kv, expected_hidden = independent_full(model, ids, return_hidden=True)
    # Seed the retained hits with true KV so even partial recompute has a dense oracle.
    for i, (latent, rope) in enumerate(expected_kv):
        caches[str(i)][0].flatten(0, 2)[:5] = latent[:5]
        caches[str(i)][1].flatten(0, 2)[:5] = rope[:5]
    stats = mla_selective_prefill(
        model, caches, ids, [0, 1], [(2, 5)], SelectiveConfig(ratio=ratio, debug_selected_positions=True),
        computed_prefix_tokens=2, output_tail_tokens=2,
    )
    assert stats["selected_positions"][-2:] == [5, 6]
    assert stats["computed_end"] == len(ids)
    if ratio == 0:
        assert stats["selected_positions"] == [5, 6]
        assert stats["retained_hit_tokens"] == 3
    torch.testing.assert_close(
        stats["tail_hidden_states"], expected_hidden[-2:], atol=3e-6, rtol=2e-5
    )
    for i, (latent, rope) in enumerate(expected_kv):
        torch.testing.assert_close(
            caches[str(i)][0].flatten(0, 2)[5:7], latent[-2:], atol=3e-6, rtol=2e-5
        )
        torch.testing.assert_close(
            caches[str(i)][1].flatten(0, 2)[5:7], rope[-2:], atol=3e-6, rtol=2e-5
        )


def test_sparse_attention_uses_absolute_positions():
    torch.manual_seed(1)
    q, k, v = torch.randn(3, 2, 8), torch.randn(9, 2, 8), torch.randn(9, 2, 3)
    positions = torch.tensor([0, 4, 8])
    expected = F.scaled_dot_product_attention(
        q.transpose(0, 1),
        k.transpose(0, 1),
        v.transpose(0, 1),
        attn_mask=torch.arange(9)[None] <= positions[:, None],
        scale=0.3,
    ).transpose(0, 1)
    torch.testing.assert_close(reference_attention(q, k, v, positions, 0.3), expected)


@pytest.mark.parametrize("invalid", ["shape", "dtype", "blocks", "ranges", "sp"])
def test_invalid_input_rejected_before_cache_mutation(invalid):
    model, caches = fixture()
    blocks, hits = [0, 1], [(1, 5)]
    if invalid == "shape":
        caches["2"] = (torch.zeros(4, 4, 1, 6), caches["2"][1])
    elif invalid == "dtype":
        caches["2"] = (caches["2"][0].half(), caches["2"][1].half())
    elif invalid == "blocks":
        blocks = [0, 0]
    elif invalid == "ranges":
        hits = [(3, 5), (2, 4)]
    else:
        model.model.layers[2].use_sequence_parallel_moe = True
    before = caches["0"][0].clone()
    with pytest.raises(ValueError):
        mla_selective_prefill(
            model, caches, list(range(7)), blocks, hits, SelectiveConfig()
        )
    torch.testing.assert_close(before, caches["0"][0], equal_nan=True)


def test_relocation_preserves_yarn_amplitude_and_content():
    from lmcache_ascend.v1.blend.mla_selective import relocation_rope_cache

    model, _ = fixture()
    rope = model.model.layers[0].self_attn.rotary_emb
    x = torch.randn(5, 1, 4)
    old, new = torch.arange(5), torch.arange(7, 12)
    cached = rotate_positions(x, old, rope)
    table = relocation_rope_cache(rope.cos_sin_cache, x.dtype)

    def complex_rotation(z, cs):
        phase = torch.complex(cs[:, :2], cs[:, 2:])[:, None]
        z = torch.view_as_complex(z.reshape(5, 1, 2, 2).contiguous())
        return z, phase

    z, phase_old = complex_rotation(cached, table[old])
    _, phase_new = complex_rotation(cached, table[new])
    relocated = torch.view_as_real(z * phase_old.conj() * phase_new).flatten(-2)
    torch.testing.assert_close(relocated, rotate_positions(x, new, rope))


def test_native_moe_context_tracks_actual_layer_registration():
    from lmcache_ascend.v1.blend.mla_selective import align_moe_context

    context = NS(
        all_moe_layers=[
            "model.layers.3.mlp.experts",
            "model.layers.4.mlp.experts",
            "model.layers.40.mlp.experts",
        ],
        moe_layer_index=0,
    )
    align_moe_context(context, "model.layers.4.self_attn.attn")
    assert context.moe_layer_index == 1
    align_moe_context(context, "model.layers.40.self_attn.attn")
    assert context.moe_layer_index == 2
    context.all_moe_layers = None
    align_moe_context(context, "model.layers.4.self_attn.attn")
    assert context.moe_layer_index == 2


@pytest.mark.parametrize("ratio", [0.0, 0.4, 1.0])
def test_debug_positions_does_not_change_mla_kv_or_tail(monkeypatch, ratio):
    model, caches = fixture()
    ids = [1, 3, 5, 7, 9, 11, 13]
    old = independent_full(model, [2, 4, 6, 8, 10, 12, 14])
    for i, (latent, rope) in enumerate(old):
        caches[str(i)][0].flatten(0, 2)[1:5] = latent[1:5]
        caches[str(i)][1].flatten(0, 2)[1:5] = rope[1:5]
    debug_caches = {name: [x.clone() for x in pair] for name, pair in caches.items()}

    def unexpected_cpu_copy(*args, **kwargs):
        pytest.fail("Default MLA stats must not copy positions to CPU")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", unexpected_cpu_copy)
        stats = mla_selective_prefill(
            model, caches, ids, [0, 1], [(1, 5)], SelectiveConfig(ratio),
            output_tail_tokens=2,
        )
    debug = mla_selective_prefill(
        model, debug_caches, ids, [0, 1], [(1, 5)],
        SelectiveConfig(ratio, debug_selected_positions=True), output_tail_tokens=2,
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
