# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""仅对有效行做 SwiGLU 的自定义 AscendC 算子封装（P3-2b）。

vllm-ascend 在 EP 场景下把 MoE 的 gate/up 投影输出扩到 NUM_ROWS*K 行，但只有
前 A=sum(group_list) 行是本 rank 本地专家的有效输出。CANN 的 SwiGlu 会对整个
padding 缓冲区做逐元素计算（实测 ~2M 行里只有 6.3% 有效）。本模块通过 aclnn
调用自研 ``aclnnMoeGroupedSwiglu``，只计算前 A 行。

开关：``VLLM_ASCEND_GROUPED_SWIGLU=1``；库路径默认指向镜像内 op 包安装位置，
可用 ``VLLM_ASCEND_GROUPED_SWIGLU_LIB`` 覆盖。任一前置条件不满足或调用失败时
返回 ``None``，由调用方回退到 ``torch_npu.npu_swiglu``。
"""

import os

import torch
from vllm.logger import logger  # 复用 vllm 命名空间，保证 worker 内 INFO 可见

_DEFAULT_LIB = (
    "/opt/ascend-custom-ops/vendors/customize/op_api/lib/libcust_opapi.so"
)
_LIB_PATH = os.environ.get("VLLM_ASCEND_GROUPED_SWIGLU_LIB", _DEFAULT_LIB)
_ENABLED = os.environ.get("VLLM_ASCEND_GROUPED_SWIGLU", "0") == "1"

_runtime = None
_disabled_reason = None
_logged_once = False


def _runtime_with_custom_lib():
    """复用 fla_npu 的 aclnn ctypes runtime，并把自定义 op_api 库插到最前。"""
    global _runtime
    if _runtime is None:
        import ctypes

        import fla_npu.ops.ascendc._runtime as fla_runtime

        lib = ctypes.CDLL(_LIB_PATH, mode=ctypes.RTLD_GLOBAL)
        rt = fla_runtime.runtime()
        rt._libraries = [lib] + list(rt._libraries)
        rt._symbols.clear()
        _runtime = rt
    return _runtime


def grouped_swiglu(
    hidden_states: torch.Tensor,
    group_list: torch.Tensor | None,
    group_list_type: int,
) -> torch.Tensor | None:
    """返回 SwiGLU 结果；不满足条件或失败时返回 None，调用方回退。"""
    global _disabled_reason, _logged_once

    if not _ENABLED or _disabled_reason is not None:
        return None
    if group_list_type != 1 or group_list is None or group_list.numel() == 0:
        return None
    if hidden_states.dim() != 2 or hidden_states.dtype != torch.bfloat16:
        return None
    if not hidden_states.is_contiguous() or hidden_states.shape[-1] % 2:
        return None
    width = hidden_states.shape[-1] // 2
    if width <= 0 or width > 4096:
        return None

    try:
        if group_list.dtype != torch.int64:
            group_list = group_list.to(torch.int64)
        import fla_npu.ops.ascendc._runtime as fla_runtime

        _runtime_with_custom_lib()
        rows = hidden_states.shape[0]
        out = torch.empty((rows, width), dtype=hidden_states.dtype, device=hidden_states.device)
        fla_runtime.call_aclnn(
            "aclnnMoeGroupedSwiglu",
            lambda ctx: [
                ctx.tensor(hidden_states),
                ctx.tensor(group_list),
                ctx.tensor(out),
            ],
            out,
        )
        if not _logged_once:
            _logged_once = True
            logger.info(
                "MoeGroupedSwiglu active: rows=%d width=%d groups=%d",
                rows,
                width,
                group_list.numel(),
            )
        return out
    except Exception as exc:  # noqa: BLE001 - 失败必须回退，不能中断推理
        _disabled_reason = repr(exc)
        logger.warning("MoeGroupedSwiglu disabled after failure: %r", exc)
        return None
