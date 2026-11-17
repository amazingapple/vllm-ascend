# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""仅对有效行做 SwiGLU 的 MoE 激活封装（P3-2b）。

vllm-ascend 在 EP 场景下把 MoE 的 gate/up 投影输出扩到 NUM_ROWS*K 行，但只有
前 A=sum(group_list) 行是本 rank 本地专家的有效输出。CANN 的 SwiGlu 会对整个
padding 缓冲区做逐元素计算（实测 ~2M 行里只有 6.3% 有效）。

本模块调用 CANN 内置 ``aclnnClippedSwiglu``（``torch_npu.npu_clipped_swiglu``）
的 group_index 模式，只计算前 A 行。``alpha=1.0, bias=0.0, limit=1e30`` 与
``torch_npu.npu_swiglu`` 逐 bit 等价（实测 66,142,208/66,142,208 一致）。

开关：``VLLM_ASCEND_GROUPED_SWIGLU=1``；``VLLM_ASCEND_GROUPED_SWIGLU_LIMIT``
可覆盖 clamp 上界（默认 1e30，即不 clamp）。任一前置条件不满足或调用失败时
返回 ``None``，由调用方回退到 ``torch_npu.npu_swiglu``。
"""

import os

import torch
import torch_npu
from vllm.logger import logger  # 复用 vllm 命名空间，保证 worker 内 INFO 可见

# 标准 SwiGLU 不做 clamp；需要与带 swiglu_limit 的激活对齐时才用环境变量打开。
_CLIP_LIMIT = float(os.environ.get("VLLM_ASCEND_GROUPED_SWIGLU_LIMIT", "1e30"))
_ENABLED = os.environ.get("VLLM_ASCEND_GROUPED_SWIGLU", "0") == "1"

_disabled_reason = None
_logged_once = False


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
        out = torch_npu.npu_clipped_swiglu(
            hidden_states,
            group_index=group_list,
            dim=-1,
            alpha=1.0,
            bias=0.0,
            limit=_CLIP_LIMIT,
            interleaved=False,
        )
        if not _logged_once:
            _logged_once = True
            logger.info(
                "MoeGroupedSwiglu active: rows=%d width=%d groups=%d",
                hidden_states.shape[0],
                width,
                group_list.numel(),
            )
        return out
    except Exception as exc:  # noqa: BLE001 - 失败必须回退，不能中断推理
        _disabled_reason = repr(exc)
        logger.warning("MoeGroupedSwiglu disabled after failure: %r", exc)
        return None
