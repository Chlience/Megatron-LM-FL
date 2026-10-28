# Copyright (c) 2026, MegaLens Authors. All rights reserved.
"""Implicit pipeline phase metadata and scopes; scheduling remains in Core."""

import torch

from megatron.core.observability import open_trace_scope, prepare_trace_scope


def _set_pipeline_operation_identity(scope, current_microbatch, vp_stage):
    """Fill operation identity only when the phase scope is active."""
    if scope.get("timing_phase") != "framework_phase":
        return
    if current_microbatch is None:
        operation_id = None
    else:
        vp_identity = "none" if vp_stage is None else vp_stage
        operation_id = f"pp:microbatch={current_microbatch}:vp={vp_identity}"
    scope.set("operation_id", operation_id)


def _pipeline_workload_from_output(
    output_tensor, config, *, cp_group_size, tp_group_size, fallback_num_tokens
):
    """Derive the source MixedPara workload from the pre-loss output shape."""
    current_num_tokens = 0
    actual_tensor = output_tensor[0] if isinstance(output_tensor, list) else output_tensor
    if isinstance(actual_tensor, torch.Tensor) and actual_tensor.ndim >= 2:
        current_num_tokens = int(actual_tensor.shape[0]) * int(actual_tensor.shape[1])
        current_num_tokens *= cp_group_size
        if getattr(config, "sequence_parallel", False):
            current_num_tokens *= tp_group_size

    if current_num_tokens == 0:
        current_num_tokens = fallback_num_tokens

    return int(current_num_tokens), float(current_num_tokens**2)


def _set_pipeline_workload(scope, workload):
    """Fill workload slots only when the phase scope is active."""
    if workload is None or scope.get("timing_phase") != "framework_phase":
        return
    num_tokens, sum_sq_seq_len = workload
    scope.set("num_tokens", num_tokens)
    scope.set("sum_sq_seq_len", sum_sq_seq_len)


def _optional_bool(value):
    """Preserve unknown schedule metadata instead of asserting a false value."""
    return None if value is None else bool(value)


def _pipeline_phase_context(*, current_microbatch, vp_stage, is_first_microbatch, is_last_stage):
    """Build phase metadata after the facade confirms that tracing is active."""
    ctx = {
        "current_microbatch": current_microbatch,
        "vp_stage": vp_stage,
        "is_first_microbatch": _optional_bool(is_first_microbatch),
        "is_last_stage": _optional_bool(is_last_stage),
        "timing_phase": "framework_phase",
    }
    return ctx


def _pipeline_phase_scope(
    name, *, current_microbatch, vp_stage, is_first_microbatch, is_last_stage, slots
):
    gate = prepare_trace_scope(name)
    ctx = None
    if gate is not None:
        ctx = _pipeline_phase_context(
            current_microbatch=current_microbatch,
            vp_stage=vp_stage,
            is_first_microbatch=is_first_microbatch,
            is_last_stage=is_last_stage,
        )
    return open_trace_scope(gate, name, ctx=ctx, slots=slots)


def _pipeline_grad_sync_context(schedule):
    """Build schedule metadata only for an enabled grad-sync probe."""
    return {"schedule": schedule, "timing_phase": "framework_phase"}


def _pipeline_grad_sync_scope(schedule):
    gate = prepare_trace_scope("grad-sync")
    ctx = _pipeline_grad_sync_context(schedule) if gate is not None else None
    return open_trace_scope(gate, "grad-sync", ctx=ctx)
