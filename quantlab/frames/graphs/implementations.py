import plotly.graph_objects as go
import polars as pl
from quantlab.frames.graphs.interface import Graph
from quantlab.frames.main import FrameBase
from quantlab.frames.graphs.design import (
    Colors,
    CustomHovers,
    get_color_map,
    get_heatmap_colorscale,
    get_marker_config,
)

class Curves(Graph):
    def setup_figure(self, formatted_data: FrameBase) -> None:
        color_map: dict[str, str] = get_color_map(assets=formatted_data.names)
        for column in formatted_data.values:
            self.figure.add_trace(
                trace=go.Scattergl( # type: ignore
                    y=column,
                    x=formatted_data.index,
                    mode="lines",
                    name=column.name,
                    line=dict(width=2, color=color_map[column.name]),
                    hovertemplate=CustomHovers.VERTICAL_DATA.value,
                )
            )


class Violins(Graph):
    def setup_figure(self, formatted_data: FrameBase) -> None:
        color_map: dict[str, str] = get_color_map(assets=formatted_data.names)
        for column in formatted_data.values:
            self.figure.add_trace(
                trace=go.Violin( # type: ignore
                    y=column,
                    name=column.name,
                    box_visible=True,
                    points=False,
                    marker=get_marker_config(color=color_map[column.name]),
                    box_line_color=Colors.WHITE,
                    hoveron="violins",
                    hoverinfo="y",
                    hovertemplate=CustomHovers.VERTICAL_DATA.value,
                )
            )


class Boxes(Graph):
    def setup_figure(self, formatted_data: FrameBase) -> None:
        color_map: dict[str, str] = get_color_map(assets=formatted_data.names)
        for column in formatted_data.values:
            self.figure.add_trace(
                trace=go.Box(
                    y=column, # type: ignore
                    name=column.name,
                    marker=get_marker_config(color=color_map[column.name]),
                    boxpoints=False,
                    hovertemplate=CustomHovers.VERTICAL_DATA.value,
                )
            )

class Histograms(Graph):
    def setup_figure(self, formatted_data: FrameBase) -> None:
        color_map: dict[str, str] = get_color_map(assets=formatted_data.names)
        for column in formatted_data.values:
            self.figure.add_trace(
                trace=go.Histogram( # type: ignore
                    x=column,
                    name=column.name,
                    marker=get_marker_config(color=color_map[column.name]),
                    hovertemplate=CustomHovers.HORIZONTAL_DATA.value,
                )
            )
        self.figure.update_layout(
            barmode="overlay"
        )


class Bars(Graph):
    def setup_figure(self, formatted_data: FrameBase) -> None:
        df: pl.DataFrame = formatted_data.data.unpivot().sort(by="value")
        names: list[str] = df.select(pl.col(name="variable")).to_series().to_list()

        color_map: dict[str, str] = get_color_map(assets=names)
        for label, value in df.iter_rows():
            self.figure.add_trace(
                trace=go.Bar(
                    x=[label],
                    y=[value],
                    name=label,
                    marker=get_marker_config(color=color_map[label]),
                    hovertemplate=CustomHovers.VERTICAL_DATA.value,
                )
            )

        self.figure.update_layout(
            xaxis=dict(showticklabels=False)
        )


class HeatMap(Graph):
    def setup_figure(self, formatted_data: FrameBase) -> None:
        color_scale: list[list[float | str]] = get_heatmap_colorscale()
        self.figure.add_trace(
            trace=go.Heatmap(
                z=formatted_data.values, # type: ignore
                x=formatted_data.names, # type: ignore
                y=formatted_data.names, # type: ignore
                showscale=False,
                colorscale=color_scale, # type: ignore
                hovertemplate=CustomHovers.HEATMAP.value,
            )
        )
        self.figure.update_layout(
            yaxis=dict(showgrid=False, autorange="reversed")
        )
        self.figure.update_yaxes(showticklabels=False, scaleanchor="x")
        self.figure.update_xaxes(showticklabels=False)
