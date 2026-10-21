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
"""EB5 / EB-Lite 在昆仑上的融合算子适配层。

三个模块分工：

* :mod:`~sglang_kunlun.eb5_adapt.eb5_kunlun_adapt` —— mHC pre/post、融合 rope、
  wo_a 低秩投影；``apply()`` 是整个适配层的入口，内部依次调用另外两个模块。
* :mod:`~sglang_kunlun.eb5_adapt.eb5_mla_adapt` —— MLA / fp8 量化 / indexer /
  flash_mla 等 attention 侧拆解，并提供通用的 ``patch_everywhere`` 注入器。
* :mod:`~sglang_kunlun.eb5_adapt.eb5_extra_adapt` —— HCA / MoE / 前后处理。

模块随 ``sglang_kunlun`` 一起打包（editable 与 wheel 安装都能直接
``from sglang_kunlun.eb5_adapt import apply``）。服务侧由插件
:mod:`sglang_kunlun.hooks.eb5_kunlun_patch` 加载；单测由 ``conftest.py`` 加载。

``EB5_KUNLUN_ADAPT_DIR`` 仍可指向一个平铺目录做实验覆盖；那种场景下三个模块
按平铺名互相导入（见各模块里的 ``_import_sibling``）。
"""

from __future__ import annotations

from . import eb5_extra_adapt, eb5_kunlun_adapt, eb5_mla_adapt
from .eb5_kunlun_adapt import apply

__all__ = [
    "apply",
    "eb5_kunlun_adapt",
    "eb5_mla_adapt",
    "eb5_extra_adapt",
]
