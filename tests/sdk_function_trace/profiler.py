from __future__ import annotations

import sys
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from types import FrameType, FunctionType
from typing import Final


class PythonProfiler:
    def __init__(self, functions: Sequence[FunctionType]) -> None:
        self._names_by_code: Final = {function.__code__: function.__name__ for function in functions}
        self._seen_frames: set[FrameType] = set()
        self.events: list[str] = []

    def __call__(self, frame: FrameType, event: str, _arg: object) -> None:
        if event != "call" or frame in self._seen_frames:
            return
        function_name: Final = self._names_by_code.get(frame.f_code)
        if function_name is None:
            return
        self._seen_frames.add(frame)
        self.events.append(function_name)


@contextmanager
def profile_python(functions: Sequence[FunctionType]) -> Generator[PythonProfiler]:
    profiler: Final = PythonProfiler(functions)
    previous: Final = sys.getprofile()
    sys.setprofile(profiler)
    try:
        yield profiler
    finally:
        sys.setprofile(previous)
