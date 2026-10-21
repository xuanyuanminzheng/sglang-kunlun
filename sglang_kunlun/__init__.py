# Copyright (c) 2026 Baidu, Inc. All rights reserved.
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
"""sglang-kunlun: OOT SGLang platform plugin for Kunlun (XPU) chips."""

# Install module-path aliases before any hook module is imported, so that
# imports written against the sglang v0.5.x layout keep resolving on
# newer sglang releases that relocated the same modules.
from sglang_kunlun.bootstrap import compat_aliases as _compat_aliases

_compat_aliases.install()

# Newer sglang imports tilelang at module level from its kernel modules;
# tilelang cannot run on Kunlun and those kernels are never selected, so
# a permissive stub is enough to let the import succeed.
from sglang_kunlun.bootstrap import optional_kernel_stubs as _optional_stubs

_optional_stubs.install_all()

from sglang_kunlun.bootstrap import _kunlun_pre_shim

__all__ = ["_kunlun_pre_shim"]
