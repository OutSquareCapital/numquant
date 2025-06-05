from quantlab.graphs.implementations import (
    Bars,
    Boxes,
    Curves,
    HeatMap,
    Histograms,
    Violins,
)
from dataclasses import dataclass
import polars as pl

@dataclass(slots=True)
class FrameVisualizer:
    _parent: pl.DataFrame
    on: str
    index: str
    values: str

    def curves(self) -> None:
        Curves(data=self._parent, index=self.index, values=self.values, on=self.on)

    def violins(self) -> None:
        Violins(data=self._parent, index=self.index, values=self.values, on=self.on)

    def histograms(self) -> None:
        Histograms(data=self._parent, index=self.index, values=self.values, on=self.on)

    def boxes(self) -> None:
        Boxes(data=self._parent, index=self.index, values=self.values, on=self.on)

    def heatmap(self) -> None:
        HeatMap(data=self._parent, index=self.index, values=self.values, on=self.on)

    def bars(self) -> None:
        Bars(data=self._parent, index=self.index, values=self.values, on=self.on)
