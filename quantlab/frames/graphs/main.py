from quantlab.frames.main import FrameBase
from quantlab.frames.graphs.implementations import (
    Bars,
    Boxes,
    Curves,
    HeatMap,
    Histograms,
    Violins,
)


class FrameVisualizer:
    def __init__(self, df: FrameBase) -> None:
        self._parent: FrameBase = df

    def curves(self) -> None:
        Curves(formatted_data=self._parent)

    def violins(self) -> None:
        Violins(formatted_data=self._parent)

    def histograms(self) -> None:
        Histograms(formatted_data=self._parent)

    def boxes(self) -> None:
        Boxes(formatted_data=self._parent)

    def heatmap(self) -> None:
        HeatMap(formatted_data=self._parent)

    def bars(self) -> None:
        Bars(formatted_data=self._parent)
