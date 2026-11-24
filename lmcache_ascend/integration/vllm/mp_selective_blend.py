# SPDX-License-Identifier: Apache-2.0
"""Qwen3 and experimental DeepSeek MLA selective blending before tail forward."""

# Standard
from dataclasses import dataclass
import json
from typing import Any

# Third Party
from lmcache.integration.vllm.mp_blend_client import (
    BlendCoverageTask,
    LMCacheMPSelectiveConnector,
)
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.token_codec import unpack_token_ids
import torch

# First Party
from lmcache_ascend.v1.blend.mp_selective import SelectiveConfig, selective_prefill
from vllm.distributed import get_tp_group

logger = init_logger(__name__)

_MERGED_WORKERS = {
    "lmcache_ascend.integration.vllm.merged_mla_worker.MergedMLABlendWorker",
    "lmcache_ascend.integration.vllm.merged_mla_worker.MergedBlendWorker",
}


@dataclass
class MergedCoverageTask(BlendCoverageTask):
    """Coverage with a terminal suffix scheduled in the same engine step."""

    tail_tokens: int


class MPSelectiveBlendConnector(LMCacheMPSelectiveConnector):
    """Complete loaded coverage with model-specific selective computation.

    Uses the standard vLLM scheduler and forward context. Requires synchronous,
    eager execution, one request, no APC/LoRA, and ND KV storage.
    DeepSeek native Ascend quantization is opt-in for accuracy reference runs.
    DeepSeek and Qwen3 can merge the terminal suffix into selective prefill.
    """

    def __init__(
        self, vllm_config: Any, role: Any, kv_cache_config: Any = None
    ) -> None:
        cfg = vllm_config
        if not cfg.model_config.enforce_eager:
            raise ValueError("MP selective blending requires enforce_eager=True")
        if cfg.cache_config.enable_prefix_caching:
            raise ValueError("MP selective blending requires prefix caching disabled")
        if cfg.scheduler_config.max_num_seqs != 1:
            raise ValueError("MP selective blending requires max_num_seqs=1")
        self.mla_reference = cfg.model_config.hf_config.model_type == "deepseek_v3"
        if cfg.lora_config is not None:
            raise ValueError("MP selective blending does not support LoRA")
        if self.mla_reference:
            self.validate_mla_config(cfg)
        elif cfg.model_config.hf_config.model_type != "qwen3":
            raise ValueError("MP selective blending requires Qwen3 or DeepSeek V3")
        elif cfg.model_config.quantization is not None:
            raise ValueError("Qwen3 selective blending does not support quantization")
        self.selective_vllm_config = cfg
        get_config = cfg.kv_transfer_config.get_from_extra_config
        self.merge_tail = get_config(
            "lmcache.mp.blend_merge_tail",
            get_config("lmcache.mp.blend_merge_mla_tail", False),
        )
        if self.merge_tail and (
            cfg.parallel_config.worker_cls not in _MERGED_WORKERS
            or cfg.scheduler_config.async_scheduling
        ):
            raise ValueError("Tail merge requires the synchronous merged blend worker")
        self.selective_config = SelectiveConfig(
            ratio=float(get_config("lmcache.mp.blend_recompute_ratio", 0.15)),
            check_layer=get_config("lmcache.mp.blend_check_layer", 1),
            selection_metric=get_config("lmcache.mp.blend_selection_metric", "k"),
            debug_selected_positions=get_config(
                "lmcache.mp.blend_debug_selected_positions", False
            ),
            refresh_check_layer=get_config(
                "lmcache.mp.blend_refresh_check_layer", False
            ),
        )
        if cfg.speculative_config is not None:
            raise ValueError(
                "MP selective blending does not support speculative decoding"
            )
        parallel = cfg.parallel_config
        if parallel.pipeline_parallel_size != 1 or parallel.data_parallel_size != 1:
            raise ValueError("MP selective blending requires one DP and PP rank")
        if str(cfg.cache_config.cache_dtype).startswith("fp8"):
            raise ValueError("MP selective blending requires an unquantized KV cache")
        if kv_cache_config is not None and len(kv_cache_config.kv_cache_groups) != 1:
            raise ValueError("MP selective blending requires one KV cache group")
        self.selective_caches: dict[str, Any] = {}
        super().__init__(vllm_config, role, kv_cache_config)
        if self.lazy_offload:
            raise ValueError("MP selective blending requires eager offload")

    @staticmethod
    def validate_mla_config(cfg: Any) -> None:
        """Require explicit reference mode and preserve unsupported-mode guards."""
        extra = cfg.kv_transfer_config.get_from_extra_config
        if not extra("lmcache.mp.mla_accuracy_reference", False):
            raise ValueError(
                "DeepSeek MLA requires explicit mla_accuracy_reference mode"
            )
        if cfg.model_config.quantization not in (None, "ascend"):
            raise ValueError("MLA reference only supports native Ascend quantization")
        if str(cfg.model_config.dtype) != "torch.bfloat16":
            raise ValueError("MLA reference requires BF16 activations")
        parallel = cfg.parallel_config
        if (
            parallel.enable_expert_parallel
            or parallel.use_sequence_parallel_moe
            or parallel.prefill_context_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
        ):
            raise ValueError(
                "MLA reference requires TP-only execution, without SP/EP/CP"
            )
        if cfg.cache_config.cache_dtype not in ("auto", "bfloat16"):
            raise ValueError("MLA reference requires unquantized BF16 KV cache")

    def run_mla_reference(
        self, model: Any, *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        """Supply a native eager context sized to each selective layer call."""
        from contextlib import contextmanager

        from vllm_ascend.ascend_forward_context import set_ascend_forward_context

        from lmcache_ascend.v1.blend.mla_selective import mla_selective_prefill
        from vllm.forward_context import get_forward_context

        @contextmanager
        def layer_context(count: int, index: int):
            with set_ascend_forward_context(
                None,
                self.selective_vllm_config,
                num_tokens=count,
                num_actual_tokens=count,
                model_instance=model,
                skip_compiled=True,
            ):
                get_forward_context().layer_idx = index
                yield

        return mla_selective_prefill(
            model, *args, layer_context=layer_context, **kwargs
        )

    def _register_blend_rope(self) -> None:
        """Declare the MLA rotation window; never rotate latent content."""
        if not self.mla_reference:
            return super()._register_blend_rope()
        from lmcache.v1.platform import resolve_kv_wrapper_factory

        from vllm.forward_context import get_forward_context

        model = get_forward_context().model_instance
        attn = model.model.layers[0].self_attn
        rope = attn.rotary_emb
        from lmcache_ascend.v1.blend.mla_selective import (
            relocation_rope_cache,
            uses_ascend_mla_rope,
        )

        dtype = next(iter(self.selective_caches.values()))[0].dtype
        self.mla_relocation_table = relocation_rope_cache(rope.cos_sin_cache, dtype)
        wrapper = resolve_kv_wrapper_factory(self.mla_relocation_table.device.type)(
            self.mla_relocation_table
        )
        self.worker_adapter.req_client.cb_register_rope(
            self.worker_adapter.instance_id,
            [wrapper],
            attn.qk_rope_head_dim,
            uses_ascend_mla_rope(rope) or bool(rope.is_neox_style),
            [0],
            [[attn.kv_lora_rank, attn.qk_rope_head_dim]],
            [attn.qk_rope_head_dim],
        ).result(timeout=self._blend_rpc_timeout)
        self._blend_rope_wrapper = wrapper
        self._blend_rope_registered = True

    def register_kv_caches(self, kv_caches: dict[str, Any]) -> None:
        """Capture worker-local paged KV, rejecting packed Ascend formats."""
        # Third Party
        import torch_npu

        for pair in kv_caches.values():
            for tensor in pair:
                if self.mla_reference and not tensor.is_contiguous():
                    raise ValueError(
                        "MLA token scatter requires contiguous cache planes"
                    )
                if torch_npu.get_npu_format(tensor) not in (0, 2):
                    raise ValueError(
                        "MP selective blending requires NPU KV format 0 or 2"
                    )
        super().register_kv_caches(kv_caches)
        self.selective_caches = kv_caches

    def build_connector_meta(self, scheduler_output: Any) -> Any:
        metadata = super().build_connector_meta(scheduler_output)
        if self.merge_tail and len(scheduler_output.num_scheduled_tokens) == 1:
            for index, task in enumerate(metadata.blend_scatter_tasks):
                if not isinstance(task, BlendCoverageTask) or not task.matches:
                    continue
                tail = scheduler_output.num_scheduled_tokens.get(task.key.request_id, 0)
                if tail > 0 and task.coverage_end + tail == len(
                    unpack_token_ids(task.key.token_bytes)
                ):
                    metadata.blend_scatter_tasks[index] = MergedCoverageTask(
                        key=task.key,
                        matches=task.matches,
                        block_ids=task.block_ids,
                        computed_prefix_tokens=task.computed_prefix_tokens,
                        prefix_hit_tokens=task.prefix_hit_tokens,
                        coverage_end=task.coverage_end,
                        tail_tokens=tail,
                    )
        return metadata

    def complete_blend(
        self, task: BlendCoverageTask, loaded: bool, forward_context: Any
    ) -> None:
        """Agree on load success, then compute every outstanding position.

        A confirmed failure on any TP rank discards all hits for this step.
        Successful ranks have already waited for their writes before entering
        this collective, so full recomputation cannot race a late scatter.
        """
        group = get_tp_group()
        first_cache = next(iter(self.selective_caches.values()))[0]
        local_success = torch.tensor(
            [int(loaded)], device=first_cache.device, dtype=torch.int32
        )
        success_count = int(group.all_reduce(local_success).item())
        all_loaded = success_count == group.world_size
        ready = task.computed_prefix_tokens
        if all_loaded:
            ready = max(ready, min(task.prefix_hit_tokens, task.coverage_end))
        stats = {"computed_end": task.coverage_end, "retained_prefix_tokens": ready}
        if ready < task.coverage_end:
            executor = (
                self.run_mla_reference if self.mla_reference else selective_prefill
            )
            tail = (
                task.tail_tokens
                if all_loaded and isinstance(task, MergedCoverageTask)
                else 0
            )
            stats = executor(
                forward_context.model_instance,
                self.selective_caches,
                unpack_token_ids(task.key.token_bytes)[: task.coverage_end + tail],
                task.block_ids[0],
                [(m.cur_st, m.cur_ed) for m in task.matches] if all_loaded else [],
                self.selective_config,
                reduce_scores=group.all_reduce,
                broadcast=group.broadcast,
                computed_prefix_tokens=ready,
                **({"output_tail_tokens": tail} if tail else {}),
            )
            if tail:
                forward_context.lmcache_merged_tail_hidden_states = stats.pop(
                    "tail_hidden_states"
                )
                stats["merged_tail_tokens"] = tail
        logger.info(
            "MP_BLEND_EVENT %s",
            json.dumps(
                {
                    "event": "selective_prefill_complete",
                    "request_id": task.key.request_id,
                    "worker_id": self.worker_adapter.worker_id,
                    "load_success_count": success_count,
                    "full_recompute_fallback": not all_loaded,
                    **stats,
                }
            ),
        )

    def unregister_kv_caches(self) -> None:
        """Release captured views with the base connector's IPC registration."""
        super().unregister_kv_caches()
        self.selective_caches = {}

    def restore_kv_caches(self) -> None:
        """Reject sleep/wake until refreshed paged views are supported."""
        raise NotImplementedError("MP selective blending does not support sleep/wake")
