import time
from typing import Optional


class Timer:
    """
    Context manager for measuring elapsed time.

    Usage:
        with Timer() as t:
            # code to time
        print(t.elapsed)  # seconds
    """

    def __init__(self):
        self._start: Optional[float] = None
        self._end: Optional[float] = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._end = time.perf_counter()

    @property
    def elapsed(self) -> float:
        if self._start is None:
            raise RuntimeError("Timer has not been started.")
        end_time = self._end if self._end is not None else time.perf_counter()
        return end_time - self._start
