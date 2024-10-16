#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

from typing import List

from math import floor, ceil
import plotly.graph_objects as go
from matplotlib.colors import LinearSegmentedColormap, to_hex
from plotly.subplots import make_subplots

# !pip install -U kaleido

# docs colors: https://github.com/plotly/plotly.py/issues/2192
# ppf(normal residuals): https://stackoverflow.com/questions/60699836/how-to-use-norm-ppf


def convert_to_datetime(df: pd.DataFrame, colnames: List[str]) -> pd.DataFrame:
    for col in colnames:
        df[col] = pd.to_datetime(df[col])
    return df


def plot_prediction(
    dt: pd.Series,
    y_pred: pd.Series,
    # lower: pd.Series,
    # upper: pd.Series,
    # lower_wide: pd.Series,
    # upper_wide: pd.Series,
    y_actual: pd.Series,
    legend_position: tuple = (0,0),
    # y_id: str,
    title: str = "",
    mode="markers",
    tickfont_size=14
    ) -> None:
    body = [
        # go.Scatter(
        #     name="Estimate + 3 Standard Deviations",
        #     x=dt,
        #     y=upper_wide,
        #     mode="lines",
        #     marker=dict(color="#E7E8F0"),
        #     line=dict(width=0),
        #     showlegend=False,
        # ),
        # go.Scatter(
        #     name="Estimate - 3 Standard Deviations",
        #     x=dt,
        #     y=lower_wide,
        #     marker=dict(color="#E7E8F0"),
        #     line=dict(width=0),
        #     mode="lines",
        #     fillcolor="#E7E8F0",
        #     fill="tonexty",
        #     showlegend=False,
        # ),
        # go.Scatter(
        #     name="Estimate + Standard Deviation",
        #     x=dt,
        #     y=upper,
        #     mode="lines",
        #     marker=dict(color="#BDC1D6"),
        #     line=dict(width=0),
        #     showlegend=False,
        # ),
        # go.Scatter(
        #     name="Estimate - Standard Deviation",
        #     x=dt,
        #     y=lower,
        #     marker=dict(color="#BDC1D6"),
        #     line=dict(width=0),
        #     mode="lines",
        #     fillcolor="#BDC1D6",
        #     fill="tonexty",
        #     showlegend=False,
        # ),
        go.Scatter(
            name="Forecast",
            x=dt,
            y=y_pred,
            mode=mode,
            # line=dict(color='rgb(225, 69, 0)', width=2),  # rgb(216, 129, 71)
            line=dict(color="#841E62", width=1.5),
            marker={"size": 6},
            opacity=0.85,
        ),
    ]

    body.append(
        go.Scatter(
            name="Actual",
            x=dt,
            y=y_actual,
            mode=mode,
            marker={"size": 6, "symbol": "diamond"},
            line=dict(color="#000000", width=1.5),  # #7BCC62 / #68b562 / #7BB562
            opacity=0.85,
        )
    )

    fig = go.Figure(body)

    fig.update_layout(
        autosize=False,
        width=900,
        height=650,
        plot_bgcolor="white",
        yaxis_title="",
        title=title,
        hovermode="x",
        legend=dict(
            bgcolor='rgba(0, 0, 0, 0)',  # Set background to transparent
            bordercolor='rgba(0, 0, 0, 0)',  # Optional: remove border
            orientation="h",
            yanchor="auto",
            x=legend_position[0],
            y=legend_position[1],
            xanchor="right",  # changed
            indentation=15,  # Increase the spacing between legend items
            font=dict(size=tickfont_size),  
            ),
    )

    fig.update_xaxes(
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
        gridcolor="lightgrey",
        tickfont_size=tickfont_size,
    )
    fig.update_yaxes(
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
        gridcolor="lightgrey",
        tickfont_size=tickfont_size,
    )

    fig.show()
    # return fig


def shade_between(
    dt: pd.Series,
    y: pd.Series,
    yaxis_label: str,
    plt_title: str,
    date_ranges: List[List[str]],
    out_filename: str = None,
) -> None:
    fig = go.Figure(
        [
            go.Scatter(
                name=yaxis_label,
                x=dt,
                y=y,
                mode="lines",
                line=dict(color="rgb(28, 40, 51)", width=2),  # rgb(216, 129, 71)
            ),
        ]
    )

    for dt_rng in date_ranges:
        fig.add_vrect(
            x0=dt_rng[0],
            x1=dt_rng[1],
            fillcolor="rgb(255, 165, 0)",
            opacity=0.25,
            line_width=0,
        )
    fig.update_layout(
        autosize=False,
        width=900,
        height=650,
        plot_bgcolor="white",
        yaxis_title=yaxis_label,
        title=plt_title,
        hovermode="x",
    )
    fig.update_xaxes(
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
        gridcolor="lightgrey",
    )
    fig.update_yaxes(
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
        gridcolor="lightgrey",
    )

    if out_filename:
        if not os.path.exists("images"):
            os.mkdir("images")
        fig.write_image(f"images/{out_filename}")

    fig.show()

def plot_corr_matrix(corr):
    # Mask to only show the lower triangle
    mask = np.tril(np.ones_like(corr, dtype=bool)) | corr.abs().le(0.1)
    melt = corr.mask(mask).melt(ignore_index=False).reset_index()
    melt["size"] = melt["value"].abs() * 500  # Scale marker size appropriately

    # Create plot
    fig, ax = plt.subplots(figsize=(16, 16))

    # Define color map and normalization for color bar
    cmap = "seismic"
    norm = plt.Normalize(-1, 1)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = plt.colorbar(sm, ax=ax)
    cbar.ax.tick_params(labelsize="x-small")

    # Scatter plot to show the squares, with size reflecting the correlation value
    sns.scatterplot(ax=ax, data=melt, x="index", y="variable", size="size",
                    hue="value", hue_norm=norm, palette=cmap,
                    marker="s", legend=False)

    # Add grid lines
    xmin, xmax = (-0.5, corr.shape[0] - 0.5)
    ymin, ymax = (-0.5, corr.shape[1] - 0.5)
    ax.vlines(np.arange(xmin, xmax + 1), ymin, ymax, lw=1, color="silver")
    ax.hlines(np.arange(ymin, ymax + 1), xmin, xmax, lw=1, color="silver")

    # Set plot limits and aspect ratio
    ax.set(aspect=1, xlim=(xmin, xmax), ylim=(ymax, ymin), xlabel="", ylabel="")
    ax.tick_params(labelbottom=False, labeltop=True)
    plt.xticks(rotation=90)

    # Annotate upper triangle with values, bold and colored based on the value
    for y in range(corr.shape[0]):
        for x in range(corr.shape[1]):
            value = corr.mask(mask).to_numpy()[y, x]
            if pd.notna(value):
                color = sm.to_rgba(value)  # Get color from colormap
                plt.text(x, y, f"{value:.2f}", size="small",
                        color=color if abs(value) < 0.5 else 'white',
                        weight='bold',  # Bold text
                        ha="center", va="center")

    plt.show()