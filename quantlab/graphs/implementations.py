import plotly.graph_objects as go
import polars as pl
from quantlab.graphs.interface import Graph
from quantlab.graphs.design import (
    Colors,
    CustomHovers,
    get_color_map,
    get_heatmap_colorscale,
    get_marker_config,
)


class Curves(Graph):
    def setup_figure(self, data: pl.DataFrame, index: pl.Series) -> None:
        color_map: dict[str, str] = get_color_map(assets=data.columns)
        for column in data.columns:
            self.figure.add_trace(
                trace=go.Scattergl( # type: ignore
                    y=data.get_column(name=column).to_numpy(),
                    x=index,
                    mode="lines",
                    name=column,
                    line=dict(width=2, color=color_map[column]),
                    hovertemplate=CustomHovers.VERTICAL_DATA.value,
                )
            )


class Violins(Graph):
    def setup_figure(self, data: pl.DataFrame, index: pl.Series) -> None:
        color_map: dict[str, str] = get_color_map(assets=data.columns)
        for column in data.columns:
            self.figure.add_trace(
                trace=go.Violin(  # type: ignore
                    y=column,
                    name=column,
                    box_visible=True,
                    points=False,
                    marker=get_marker_config(color=color_map[column]),
                    box_line_color=Colors.WHITE,
                    hoveron="violins",
                    hoverinfo="y",
                    hovertemplate=CustomHovers.VERTICAL_DATA.value,
                )
            )


class Boxes(Graph):
    def setup_figure(self, data: pl.DataFrame, index: pl.Series) -> None:
        color_map: dict[str, str] = get_color_map(assets=data.columns)
        for column in data.columns:
            self.figure.add_trace(
                trace=go.Box(
                    y=data.get_column(name=column).to_numpy(),
                    name=column,
                    marker=get_marker_config(color=color_map[column]),
                    boxpoints=False,
                    hovertemplate=CustomHovers.VERTICAL_DATA.value,
                )
            )


class Histograms(Graph):
    def setup_figure(self, data: pl.DataFrame, index: pl.Series) -> None:
        color_map: dict[str, str] = get_color_map(assets=data.columns)
        for column in data.columns:
            self.figure.add_trace(
                trace=go.Histogram( # type: ignore
                    x=data.get_column(name=column).to_numpy(),
                    name=column,
                    marker=get_marker_config(color=color_map[column]),
                    hovertemplate=CustomHovers.HORIZONTAL_DATA.value,
                )
            )
        self.figure.update_layout(barmode="overlay")


class Bars(Graph):
    def setup_figure(self, data: pl.DataFrame, index: pl.Series) -> None:
        color_map: dict[str, str] = get_color_map(assets=data.columns)
        for label, value in data.iter_rows():
            self.figure.add_trace(
                trace=go.Bar(
                    x=[label],
                    y=[value],
                    name=label, 
                    marker=get_marker_config(color=color_map[label]),
                    hovertemplate=CustomHovers.VERTICAL_DATA.value,
                )
            )

        self.figure.update_layout(xaxis=dict(showticklabels=False))


class HeatMap(Graph):
    def setup_figure(self, data: pl.DataFrame, index: pl.Series) -> None:
        color_scale: list[list[float | str]] = get_heatmap_colorscale()
        self.figure.add_trace(
            trace=go.Heatmap(
                z=data.to_numpy(),
                x=data.columns, # type: ignore
                y=data.columns, # type: ignore
                showscale=False,
                colorscale=color_scale,  # type: ignore
                hovertemplate=CustomHovers.HEATMAP.value,
            )
        )
        self.figure.update_layout(yaxis=dict(showgrid=False, autorange="reversed"))
        self.figure.update_yaxes(showticklabels=False, scaleanchor="x")
        self.figure.update_xaxes(showticklabels=False)
