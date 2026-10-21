# Adapted from sgl-project/sglang (https://github.com/sgl-project/sglang)
# Copyright 2023-2024 SGLang Team
#
# This file has been modified by Baidu, Inc. to support Kunlun XPU.
# Modifications Copyright (c) 2026 Baidu, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Kunlun guard for ``SparsePrefillChunkCache.ensure_compressed``.

Upstream ends the gather setup with ``flat_ids.clamp_min_(0)``. When every
request in the chunk is shorter than one compressed page (typical for the
short prompts used in smoke tests, where ``seq_len // compress_ratio == 0``
for all rows), ``flat_ids`` is empty and the XPU ``clamp`` kernel rejects the
zero-element launch with ``INVALID-MEMORY``, killing the scheduler.

Short-circuit that case in the plugin: the gather layout is fully determined
(no compressed rows, so only the SWA bases matter) and upstream's cache is
populated with the same value it would have computed.
"""

from __future__ import annotations

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


@plugin_hook(
    "sglang.srt.layers.attention.dsv4.sparse_prefill_utils"
    ".SparsePrefillChunkCache.ensure_compressed",
    type=HookType.AROUND,
)
def ensure_compressed_kunlun(
    original_fn, self, compress_ratio, page_table, c_page_size
):
    if compress_ratio in self.compressed:
        return original_fn(self, compress_ratio, page_table, c_page_size)

    seq_lens_cpu = self.seq_lens_cpu
    if seq_lens_cpu is None:
        seq_lens_cpu = self.seq_lens.tolist()
    if any(int(length) // compress_ratio for length in seq_lens_cpu):
        return original_fn(self, compress_ratio, page_table, c_page_size)

    # Inline import keeps registration free of the upstream module import.
    from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import (
        CompressedGather,
    )

    device = self.seq_lens.device
    zeros = torch.zeros(self.num_reqs, dtype=torch.int32, device=device)
    gather = CompressedGather(
        flat_token_ids=torch.empty(0, dtype=torch.int32, device=device),
        compressed_base=zeros,
        swa_base=0 + self.swa_offsets[:-1],
    )
    self.compressed[compress_ratio] = gather
    return gather
