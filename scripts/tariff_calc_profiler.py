"""tariff_calc_profiler.py – tracks tariff calculation durations."""


class TariffCalcProfiler:
    def __init__(self):
        self._samples: list[float] = []

    def record(self, ms: float) -> None:
        self._samples.append(ms)

    def avg_ms(self) -> float:
        return sum(self._samples) / len(self._samples) if self._samples else 0.0
