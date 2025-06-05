from quantlab.types import ArrayBase
from quantlab.graphs.implementations import (
    Bars,
    Boxes,
    Curves,
    HeatMap,
    Histograms,
    Violins,
)
from dataclasses import dataclass


@dataclass(slots=True)
class FrameVisualizer:
    _parent: ArrayBase

    def curves(self) -> None:
        Curves(data=self._parent)

    def violins(self) -> None:
        Violins(data=self._parent)

    def histograms(self) -> None:
        Histograms(data=self._parent)

    def boxes(self) -> None:
        Boxes(data=self._parent)

    def heatmap(self) -> None:
        HeatMap(data=self._parent)

    def bars(self) -> None:
        Bars(data=self._parent)
