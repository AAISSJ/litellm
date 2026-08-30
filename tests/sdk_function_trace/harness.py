from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import FunctionType
from typing import Final

from tests.sdk_function_trace.profiler import profile_python


@dataclass(frozen=True, slots=True)
class TraceScenario:
    functions: tuple[FunctionType, ...]
    invoke_python: Callable[[], object]
    invoke_rust: Callable[[], Sequence[str]]


def assert_function_trace_parity(scenario: TraceScenario) -> None:
    expected: Final = tuple(function.__name__ for function in scenario.functions)
    with profile_python(scenario.functions) as profiler:
        scenario.invoke_python()
    python_trace: Final = tuple(profiler.events)
    rust_trace: Final = tuple(scenario.invoke_rust())

    if python_trace != expected:
        raise AssertionError(f"Python function trace differs: {python_trace!r} != {expected!r}")
    if rust_trace != expected:
        raise AssertionError(f"Rust function trace differs: {rust_trace!r} != {expected!r}")
    if python_trace != rust_trace:
        raise AssertionError(f"Python and Rust function traces differ: {python_trace!r} != {rust_trace!r}")
