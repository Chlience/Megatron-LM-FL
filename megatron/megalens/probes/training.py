# Copyright (c) 2026, MegaLens Authors. All rights reserved.
"""Implicit training lifecycle calls without importing the tracing runtime."""

import functools
import logging
from contextlib import AbstractContextManager, nullcontext
from typing import Any, Callable, ParamSpec, TypeVar

import torch

_P = ParamSpec('_P')
_R = TypeVar('_R')


def owns_runtime(shutdown: Callable[..., None]) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Close the runtime after pretrain while preserving its result or failure."""

    def decorate(pretrain_func: Callable[_P, _R]) -> Callable[_P, _R]:
        @functools.wraps(pretrain_func)
        def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            try:
                result = pretrain_func(*args, **kwargs)
            except BaseException:
                try:
                    shutdown(graceful=False)
                except BaseException:
                    logging.getLogger(pretrain_func.__module__).exception(
                        "MegaLens non-graceful shutdown failed while handling a pretrain failure"
                    )
                raise
            shutdown(graceful=True)
            return result

        return wrapped

    return decorate


def iteration_scope(
    args: Any, iteration: int, get_runtime: Callable[[], Any], *, skip_iteration: bool
) -> AbstractContextManager[None]:
    """Prepare the existing iteration envelope and mode-1 WORLD barrier."""
    if not getattr(args, 'trace', False):
        return nullcontext()

    runtime = get_runtime()
    if skip_iteration:
        runtime.tracer.close_trace_window()
        return nullcontext()
    if not runtime.tracer.is_mode0():
        # Preserve source mode-1 alignment before entering the iteration scope.
        torch.distributed.barrier()
    return runtime.iteration(
        iteration + 1, enable_hw_monitor=getattr(args, 'hardware_monitor', False)
    )
