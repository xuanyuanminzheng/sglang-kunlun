# Copyright (c) 2026 Baidu, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Kunlun guard for zero-token embedding calls.

DP-attention idle ranks run a forward with an empty batch every step. The XPU
embedding kernel rejects a zero-element grid, so return an empty tensor instead
of launching it.
"""

from __future__ import annotations

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


@plugin_hook(
    "sglang.srt.layers.vocab_parallel_embedding.VocabParallelEmbedding"
    "._embed_local_shard",
    type=HookType.AROUND,
)
def embed_local_shard_kunlun(original_fn, self, input_):
    if input_.numel() == 0:
        return self.weight.new_empty(tuple(input_.shape) + (self.weight.shape[1],))
    return original_fn(self, input_)
