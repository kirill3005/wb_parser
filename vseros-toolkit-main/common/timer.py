import time
from typing import Dict, Optional


class StageTimer:
    def __init__(self) -> None:
        self.started_at: Dict[str, float] = {}
        self.durations: Dict[str, float] = {}

    def start(self, name: str) -> None:
        self.started_at[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        if name not in self.started_at:
            raise KeyError(f"Stage '{name}' was not started")
        elapsed = time.perf_counter() - self.started_at.pop(name)
        self.durations[name] = elapsed
        return elapsed

    def __enter__(self):
        self.start("total")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop("total")

    def get(self, name: str) -> Optional[float]:
        return self.durations.get(name)
