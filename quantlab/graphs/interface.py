from abc import ABC, abstractmethod
from quantlab.graphs.design import FigureSetup, Colors
import plotly.graph_objects as go
import polars as pl

class Graph(ABC):
    def __init__(self, data: pl.DataFrame, on: str, index: str, values: str) -> None:
        df: pl.DataFrame = data.pivot(
            on=on,
            index=index,
            values=values)
        idx: pl.Series = df.get_column(name=index)
        self.figure = go.Figure()
        self.setup_figure(data=df.drop(index), index=idx)
        self._setup_general_design()
        self._setup_axes()
        self.figure.show()

    @abstractmethod
    def setup_figure(self, data: pl.DataFrame, index: pl.Series) -> None:
        raise NotImplementedError

    def _setup_general_design(self) -> None:
        self.figure.update_layout(
            font=FigureSetup.TEXT_FONT.value,
            autosize=True,
            margin=dict(l=30, r=30, t=40, b=30),
            paper_bgcolor=Colors.BLACK,
            plot_bgcolor=Colors.BLACK,
            legend={
                "title_font": FigureSetup.LEGEND_TITLE_FONT.value,
            },
        )

    def _setup_axes(self) -> None:
        self.figure.update_yaxes(
            showgrid=False, automargin=True
        )

        self.figure.update_xaxes(
            showgrid=False, automargin=True
        )
