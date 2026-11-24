# SPDX-License-Identifier: Apache-2.0
"""Use completed selective hidden states in the Ascend V1 runner."""

from functools import wraps

from vllm.forward_context import get_forward_context
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
from vllm_ascend.worker.worker import NPUWorker


class MergedBlendWorker(NPUWorker):
    """Keep the runner's logits, sampler and connector finalization paths."""

    def init_device(self):
        if self.use_v2_model_runner:
            raise ValueError("Tail merge requires the Ascend V1 runner")
        super().init_device()
        runner = self.model_runner
        if not isinstance(runner, NPUModelRunner):
            raise TypeError("Tail merge requires an Ascend NPUModelRunner")
        model_forward = runner._model_forward

        @wraps(model_forward)
        def forward(num_tokens_padded, *args, **kwargs):
            context = get_forward_context()
            hidden = getattr(context, "lmcache_merged_tail_hidden_states", None)
            if hidden is None:
                return model_forward(num_tokens_padded, *args, **kwargs)
            del context.lmcache_merged_tail_hidden_states
            if hidden.shape[0] != num_tokens_padded:
                raise RuntimeError("Merged output does not match the scheduled tail")
            return hidden

        runner._model_forward = forward


MergedMLABlendWorker = MergedBlendWorker
