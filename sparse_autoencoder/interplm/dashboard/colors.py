from typing import List, Tuple

import numpy as np
import plotly.graph_objects as go


def generate_discrete_palette(
    values: Tuple[float, float, float], colors: Tuple[str, str, str], n: int
) -> Tuple[List[str], np.ndarray]:
    low, median, high = values
    low_color, mid_color, high_color = colors

    lower_range = np.linspace(low, median, n)[:-1]
    upper_range = np.linspace(median, high, n)[1:]

    full_range = np.concatenate([lower_range, [median], upper_range])

    lower_colors = [
        interpolate_color(low_color, mid_color, i / (n - 1)) for i in range(n - 1)
    ]
    upper_colors = [
        interpolate_color(mid_color, high_color, i / (n - 1)) for i in range(1, n)
    ]

    full_palette = lower_colors + [mid_color] + upper_colors

    return full_palette, full_range


def get_colormap_fn(palette: List[str], ranges: np.ndarray):
    def colormap_fn(value: float) -> str:
        index = np.searchsorted(ranges, value, side="right") - 1
        return palette[max(0, min(index, len(palette) - 1))]

    return colormap_fn


def interpolate_color(color1: str, color2: str, t: float) -> str:
    rgb1 = [int(color1[i: i + 2], 16) for i in (1, 3, 5)]
    rgb2 = [int(color2[i: i + 2], 16) for i in (1, 3, 5)]
    rgb = [int(round((1 - t) * c1 + t * c2)) for c1, c2 in zip(rgb1, rgb2)]
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def visualize_palette(palette: List[str], ranges: np.ndarray):
    fig = go.Figure()

    for i, color in enumerate(palette):
        fig.add_trace(
            go.Bar(
                x=[1],
                y=[1],
                marker_color=color,
                text=[f"{ranges[i]:.2f}"],
                textposition="inside",
                hoverinfo="none",
                showlegend=False,
                opacity=0.9,
            )
        )

    fig.update_layout(
        xaxis_title="",
        yaxis_title="Color Index",
        barmode="stack",
        height=300,
        width=25,
        yaxis={"visible": False},
        xaxis={"visible": False},
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )

    return fig


def select_color_range(values, feature: int | None = None) -> Tuple[int, int, int]:

    # Select the feature values if using feature-based color selection
    if feature is None:
        values = values.flatten()
    else:
        values = values[:, feature]

    min_value = np.min(values)

    # Identify the median and top 5% values of the non-zero values
    values = values[values != 0]
    median_value = np.quantile(values, 0.5)
    top_value = np.quantile(values, 0.95)

    return min_value, median_value, top_value


def get_structure_palette_and_colormap(color_range):
    cyan, white, magenta = "#00DDFF", "#FFFFFF", "#cc39ca"
    structure_color_pallete, structure_color_range = generate_discrete_palette(
        values=color_range, colors=(cyan, white, magenta), n=10
    )
    structure_colormap_fn = get_colormap_fn(
        structure_color_pallete, structure_color_range
    )
    palette_to_viz = visualize_palette(
        structure_color_pallete, structure_color_range
    )
    return structure_colormap_fn, palette_to_viz
