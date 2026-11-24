# SPDX-License-Identifier: Apache-2.0
"""Experimental DeepSeek MLA selective prefill for accuracy evaluation.

Uses the loaded projection/MoE modules (including their quantization methods).
Ascend MLA uses native fused attention and explicit sparse-position masks.
CPU contracts use a query-chunked FP32 oracle. This is not performance-tuned.
Only BF16/FP32 ND caches with separate latent and RoPE planes are accepted.
A native-forward equivalence gate is required before reporting model accuracy.
"""

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import torch

from lmcache_ascend.v1.blend.mp_selective import (
    SelectiveConfig,
    TensorOp,
    identity,
    select_positions,
)


def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    positions: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Causal MLA reference with absolute positions and bounded score memory."""
    output = torch.empty((*q.shape[:2], v.shape[-1]), device=q.device, dtype=q.dtype)
    keys = torch.arange(k.shape[0], device=q.device)
    kh = k.transpose(0, 1).float().transpose(1, 2)
    vh = v.transpose(0, 1).float()
    for start in range(0, q.shape[0], 64):
        end = min(start + 64, q.shape[0])
        scores = torch.bmm(q[start:end].transpose(0, 1).float(), kh) * scale
        scores.masked_fill_(
            keys[None, None] > positions[None, start:end, None], -torch.inf
        )
        output[start:end] = (
            torch.bmm(scores.softmax(-1), vh).transpose(0, 1).to(q.dtype)
        )
    return output


def mla_attention(attn, q, k, v, positions):
    """Match native MLA attention precision; mask sparse absolute positions."""
    if not uses_ascend_mla_rope(attn.rotary_emb):
        return reference_attention(q, k, v, positions, attn.scaling)
    import torch_npu

    full = q.shape[0] == k.shape[0]
    if full:
        mask = torch.ones(2048, 2048, device=q.device, dtype=torch.bool).triu(1)
    else:
        mask = (
            torch.arange(k.shape[0], device=q.device)[None] > positions[:, None]
        )[None, None]
    nope = attn.qk_nope_head_dim
    output, _ = torch_npu.npu_fused_infer_attention_score(
        q[..., :nope],
        k[..., :nope].contiguous(),
        v.contiguous(),
        query_rope=q[..., nope:],
        key_rope=k[..., nope:].contiguous(),
        num_heads=attn.num_local_heads,
        num_key_value_heads=attn.num_local_heads,
        input_layout="TND",
        atten_mask=mask,
        sparse_mode=3 if full else 0,
        scale=attn.scaling,
        antiquant_mode=0,
        antiquant_scale=None,
        block_table=None,
        block_size=0,
        softmax_lse_flag=True,
        actual_seq_lengths=[q.shape[0]],
        actual_seq_lengths_kv=[k.shape[0]],
    )
    return output


def align_moe_context(context: Any, attention_name: str) -> None:
    """Select the actual layer in vLLM's ordered MoE custom-op registry."""
    if context.all_moe_layers is None:
        return
    prefix = attention_name.split(".self_attn.", 1)[0] + "."
    matches = [
        i for i, name in enumerate(context.all_moe_layers) if name.startswith(prefix)
    ]
    if len(matches) > 1:
        raise ValueError("MLA reference requires one MoE registration per layer")
    if matches:
        context.moe_layer_index = matches[0]


def uses_ascend_mla_rope(rope: Any) -> bool:
    """Ascend DeepSeek stores GPT-J inputs in a NeoX output/cache layout."""
    return type(rope).__name__ == "AscendDeepseekScalingRotaryEmbedding"


def rotate_positions(
    tensor: torch.Tensor, positions: torch.Tensor, rope: Any
) -> torch.Tensor:
    """Apply the model's scaled RoPE table without global forward metadata.

    The table already includes YaRN amplitude scaling. GPT-J interleaving and
    NeoX half-splitting are both supported; only the RoPE component is passed.
    """
    width = tensor.shape[-1]
    if width != rope.head_size or width % 2:
        raise ValueError("MLA RoPE dimension does not match the model table")
    if uses_ascend_mla_rope(rope):
        # Native MLA converts interleaved projection coordinates into split
        # halves and keeps that layout in both Q and the paged RoPE cache.
        # Use the native operation to preserve its BF16 rounding as well.
        import torch_npu

        cos = rope.cos_cached[positions][:, None, None]
        sin = rope.sin_cached[positions][:, None, None]
        return torch_npu.npu_interleave_rope(
            tensor.unsqueeze(2), cos, sin
        ).squeeze(2)
    table = rope.cos_sin_cache[positions].to(tensor.dtype)
    cos, sin = table.split(width // 2, dim=-1)
    cos, sin = cos[:, None], sin[:, None]
    if rope.is_neox_style:
        a, b = tensor.chunk(2, -1)
        return torch.cat((a * cos - b * sin, b * cos + a * sin), -1)
    a, b = tensor[..., ::2], tensor[..., 1::2]
    return torch.stack((a * cos - b * sin, b * cos + a * sin), -1).flatten(-2)


def relocation_rope_cache(cache: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Remove constant YaRN amplitude from the undo/redo rotation table.

    A cached key already contains the model's amplitude. Applying the scaled
    table twice during relocation would multiply it by that amplitude squared.
    The transfer kernel implements orthogonal rotations, so register unit
    cos/sin pairs. Position-dependent amplitude is not supported by this path.
    """
    cos, sin = cache.float().chunk(2, -1)
    norm = torch.sqrt(cos.square() + sin.square())
    if not bool(torch.isfinite(norm).all()) or float(norm.min()) <= 0:
        raise ValueError("Invalid MLA RoPE table")
    if float(norm.max() / norm.min()) > 1.02:
        raise ValueError("MLA relocation requires position-independent RoPE amplitude")
    return torch.cat((cos / norm, sin / norm), -1).to(dtype).contiguous()


def project_mla(
    attn: Any, hidden: torch.Tensor, positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Use native quantized projections; return Q, normalized latent and K-rope."""
    combined = attn.fused_qkv_a_proj(hidden)[0]
    query_low, kv = combined.split(
        [attn.q_lora_rank, attn.kv_lora_rank + attn.qk_rope_head_dim], -1
    )
    query = attn.q_b_proj(attn.q_a_layernorm(query_low))[0].view(
        -1, attn.num_local_heads, attn.qk_head_dim
    )
    latent, key_rope = kv.split([attn.kv_lora_rank, attn.qk_rope_head_dim], -1)
    if uses_ascend_mla_rope(attn.rotary_emb):
        import torch_npu

        count, bs = hidden.shape[0], 128
        blocks = (count + bs - 1) // bs
        ck = hidden.new_empty((blocks, bs, 1, attn.kv_lora_rank))
        cr = hidden.new_empty((blocks, bs, 1, attn.qk_rope_head_dim))
        _, _, key_rope, latent = torch_npu.npu_kv_rmsnorm_rope_cache(
            kv.reshape(count, 1, 1, -1),
            attn.kv_a_layernorm.weight,
            attn.rotary_emb.cos_cached[positions][:, None, None],
            attn.rotary_emb.sin_cached[positions][:, None, None],
            torch.arange(count, device=hidden.device),
            cr,
            ck,
            epsilon=attn.kv_a_layernorm.variance_epsilon,
            cache_mode="PA",
            is_output_kv=True,
        )
        latent = latent.reshape(count, attn.kv_lora_rank)
        key_rope = key_rope.reshape(count, 1, attn.qk_rope_head_dim)
    else:
        latent = attn.kv_a_layernorm(latent.contiguous())
        key_rope = rotate_positions(key_rope[:, None], positions, attn.rotary_emb)
    query_nope, query_rope = query.split(
        [attn.qk_nope_head_dim, attn.qk_rope_head_dim], -1
    )
    query_rope = rotate_positions(query_rope, positions, attn.rotary_emb)
    return torch.cat((query_nope, query_rope), -1), latent, key_rope[:, 0]


def expand_mla(
    attn: Any, latent: torch.Tensor, key_rope: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand cached latent through the loaded KV up-projection for prefill."""
    expanded = attn.kv_b_proj(latent)[0].view(
        -1, attn.num_local_heads, attn.qk_nope_head_dim + attn.v_head_dim
    )
    key, value = expanded.split([attn.qk_nope_head_dim, attn.v_head_dim], -1)
    key = torch.cat((key, key_rope[:, None].expand(-1, attn.num_local_heads, -1)), -1)
    return key, value


def validate_mla_layers(model: Any, caches: Mapping[str, Any]) -> list[Any]:
    """Reject unsupported layouts before any cache is modified."""
    layers = list(model.model.layers)
    for layer in layers:
        attn = layer.self_attn
        if layer.use_sequence_parallel_moe:
            raise ValueError("MLA reference requires sequence-parallel MoE disabled")
        if attn.q_lora_rank is None:
            raise ValueError(
                "MLA reference requires DeepSeek low-rank query projection"
            )
        if attn.fused_qkv_a_proj.weight is None:
            raise ValueError(
                "MLA prolog discarded projection weights; disable fused MLAPO"
            )
        pair = caches[attn.mla_attn.mla_attn.layer_name]
        if len(pair) != 2:
            raise ValueError("MLA reference requires latent and RoPE cache planes")
        for cache, width in zip(
            pair, (attn.kv_lora_rank, attn.qk_rope_head_dim), strict=True
        ):
            if (
                cache.ndim != 4
                or cache.shape[2:] != (1, width)
                or not cache.is_contiguous()
                or cache.dtype not in (torch.bfloat16, torch.float32)
            ):
                raise ValueError(
                    "MLA reference requires contiguous BF16/FP32 [NB,BS,1,D]"
                )
        if pair[0].shape[:2] != pair[1].shape[:2]:
            raise ValueError("MLA cache planes have different block geometry")
        if pair[0].dtype != pair[1].dtype or pair[0].device != pair[1].device:
            raise ValueError("MLA cache planes must share dtype and device")
    return layers


@torch.inference_mode()
def mla_selective_prefill(
    model: Any,
    caches: Mapping[str, Sequence[torch.Tensor]],
    token_ids: Sequence[int],
    block_ids: Sequence[int],
    hit_ranges: Sequence[tuple[int, int]],
    config: SelectiveConfig,
    reduce_scores: TensorOp = identity,
    broadcast: TensorOp = identity,
    computed_prefix_tokens: int = 0,
    layer_context: Callable[[int, int], AbstractContextManager] = lambda count, index: (
        nullcontext()
    ),
    output_tail_tokens: int = 0,
) -> dict[str, Any]:
    """Refresh paged MLA KV using expanded-K/V deviation for selection.

    Selection compares reconstructed K/V, not latent-space distances. Every
    miss is mandatory. Sparse queries retain their original causal positions.
    Prefix and unselected hit cache rows remain unchanged. This is a reference
    implementation; its latency must not be presented as optimized CacheBlend.
    """
    layers = validate_mla_layers(model, caches)
    length = len(token_ids)
    if not 0 < config.check_layer < len(layers):
        raise ValueError("check_layer must lie inside the model")
    if not 0 <= computed_prefix_tokens < length:
        raise ValueError("computed prefix must lie inside the request")
    if not 0 <= output_tail_tokens <= length - computed_prefix_tokens:
        raise ValueError("output tail must follow the computed prefix")
    tail_start = length - output_tail_tokens
    previous_end = computed_prefix_tokens
    for start, end in hit_ranges:
        if not previous_end <= start < end <= tail_start:
            raise ValueError("hit ranges must be sorted, disjoint and after prefix")
        previous_end = end
    first = caches[layers[0].self_attn.mla_attn.mla_attn.layer_name][0]
    device = first.device
    block_size = first.shape[1]
    for layer in layers:
        ck = caches[layer.self_attn.mla_attn.mla_attn.layer_name][0]
        if ck.shape[1] != block_size or ck.device != device:
            raise ValueError("MLA reference requires uniform layer block geometry")
        if (
            len(block_ids) * block_size < length
            or min(block_ids) < 0
            or max(block_ids) >= ck.shape[0]
            or len(set(block_ids)) != len(block_ids)
        ):
            raise ValueError("invalid or aliased allocated cache blocks")
    positions = torch.arange(length, device=device)
    block_table = torch.tensor(block_ids, device=device, dtype=torch.long)
    slots = block_table[positions // block_size] * block_size + positions % block_size
    hits = torch.tensor(
        [p for start, end in hit_ranges for p in range(start, end)],
        device=device,
        dtype=torch.long,
    )
    prefix = torch.arange(computed_prefix_tokens, device=device)
    positions = positions[computed_prefix_tokens:]
    selected = positions
    hidden = model.embed_input_ids(
        torch.tensor(token_ids[computed_prefix_tokens:], device=device)
    )
    residual = None
    query_counts, projection_counts = [], []
    for index, layer in enumerate(layers):
        attn = layer.self_attn
        ck, cr = caches[attn.mla_attn.mla_attn.layer_name]
        paged_c, paged_r = (
            ck.view(-1, attn.kv_lora_rank),
            cr.view(-1, attn.qk_rope_head_dim),
        )
        full_c = ck.new_empty((length, attn.kv_lora_rank))
        full_r = cr.new_empty((length, attn.qk_rope_head_dim))
        if residual is None:
            residual = hidden
            hidden = layer.input_layernorm(hidden)
        else:
            hidden, residual = layer.input_layernorm(hidden, residual)
        projection_counts.append(hidden.shape[0])
        with layer_context(hidden.shape[0], index):
            q, latent, rope = project_mla(attn, hidden, positions)
        if prefix.numel():
            full_c[prefix], full_r[prefix] = (
                paged_c[slots[prefix]],
                paged_r[slots[prefix]],
            )
        if index >= config.check_layer and hits.numel():
            full_c[hits], full_r[hits] = paged_c[slots[hits]], paged_r[slots[hits]]
        if index == config.check_layer:
            # Compare only valid hit rows; never expand uninitialized cache gaps.
            local_hits = hits - computed_prefix_tokens
            fresh = cached = latent
            if hits.numel() and config.ratio > 0:
                metric_width = attn.num_local_heads * (
                    attn.qk_head_dim if config.selection_metric == "k" else attn.v_head_dim
                )
                fresh = latent.new_zeros((length - computed_prefix_tokens, metric_width))
                cached = torch.zeros_like(fresh)
                fresh_k, fresh_v = expand_mla(
                    attn, latent[local_hits], rope[local_hits]
                )
                cached_k, cached_v = expand_mla(attn, full_c[hits], full_r[hits])
                fresh[local_hits] = (
                    fresh_k if config.selection_metric == "k" else fresh_v
                ).flatten(1)
                cached[local_hits] = (
                    cached_k if config.selection_metric == "k" else cached_v
                ).flatten(1)
            chosen = select_positions(
                fresh, cached, local_hits, config.ratio, reduce_scores, broadcast
            )
            if config.refresh_check_layer:
                full_c[positions], full_r[positions] = latent, rope
            positions = chosen + computed_prefix_tokens
            selected = positions
            q, latent, rope, residual = (
                q[chosen],
                latent[chosen],
                rope[chosen],
                residual[chosen],
            )
        full_c[positions], full_r[positions] = latent, rope
        # Before selection all suffix rows are projected; after it retained hits
        # plus mandatory misses cover every suffix slot.
        k, v = expand_mla(attn, full_c, full_r)
        attended = mla_attention(attn, q, k, v, positions)
        with layer_context(positions.numel(), index):
            hidden = attn.o_proj(attended.flatten(1), is_prefill=True)[0]
            hidden, residual = layer.post_attention_layernorm(hidden, residual)
            if uses_ascend_mla_rope(attn.rotary_emb):
                from vllm.forward_context import get_forward_context

                align_moe_context(
                    get_forward_context(), attn.mla_attn.mla_attn.layer_name
                )
            hidden = layer.mlp(hidden)
        write_positions = (
            torch.arange(computed_prefix_tokens, length, device=device)
            if index < config.check_layer
            or (index == config.check_layer and config.refresh_check_layer)
            else positions
        )
        paged_c[slots[write_positions]] = full_c[write_positions]
        paged_r[slots[write_positions]] = full_r[write_positions]
        query_counts.append(positions.numel())
    # Shape metadata gives counts without a device-to-host synchronization.
    selected_count = selected.numel()
    mandatory = length - computed_prefix_tokens - hits.numel()
    stats = {
        "computed_end": length,
        "check_layer": config.check_layer,
        "recompute_ratio": config.ratio,
        "selection_metric": "expanded_" + config.selection_metric,
        "selected_hit_tokens": selected_count - mandatory,
        "retained_hit_tokens": hits.numel() - (selected_count - mandatory),
        "mandatory_tokens": mandatory,
        "retained_prefix_tokens": computed_prefix_tokens,
        "layer_qkv_counts": projection_counts,
        "layer_query_counts": query_counts,
        "implementation": "mla_accuracy_reference",
    }
    if config.debug_selected_positions:
        stats["selected_positions"] = selected.cpu().tolist()
    if output_tail_tokens:
        # Selection emits sorted positions and retains all misses. The terminal
        # suffix is outside every hit range, so its rows are contiguous here.
        tail = hidden[-output_tail_tokens:]
        tail_residual = residual[-output_tail_tokens:]
        stats["tail_hidden_states"], _ = model.model.norm(tail, tail_residual)
    return stats
