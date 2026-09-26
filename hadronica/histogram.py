"""Fixed-bin histogram used by the event display and the tests."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Histogram:
    lo: float
    hi: float
    bins: int
    counts: list[float] = field(init=False)
    entries: int = 0
    total: float = 0.0

    def __post_init__(self) -> None:
        if self.bins < 1 or self.hi <= self.lo:
            raise ValueError("histogram needs a positive span and at least one bin")
        self.counts = [0.0] * self.bins

    @property
    def width(self) -> float:
        return (self.hi - self.lo) / self.bins

    def fill(self, value: float, weight: float = 1.0) -> None:
        if value < self.lo or value >= self.hi:
            return
        index = int((value - self.lo) / self.width)
        index = min(index, self.bins - 1)
        self.counts[index] += weight
        self.entries += 1
        self.total += weight

    def clear(self) -> None:
        self.counts = [0.0] * self.bins
        self.entries = 0
        self.total = 0.0

    def mean(self) -> float | None:
        if self.total <= 0.0:
            return None
        acc = 0.0
        for i, count in enumerate(self.counts):
            center = self.lo + (i + 0.5) * self.width
            acc += center * count
        return acc / self.total
