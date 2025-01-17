"""Plotting utilities."""

import numpy as np
import pandas as pd

import matplotlib.colors as mc
import colorsys

import plotly.graph_objects as go
from plotly.graph_objects import Figure
from plotly.subplots import make_subplots


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Credit: https://stackoverflow.com/a/49601444

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def init_profile_fig(title=None, y_titles=None) -> Figure:

    # create plot
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.update_layout(
        xaxis_title='Day of Year & Hour',
        xaxis=dict(
            type="date",
            tickformat='%j %H:%M', # day of year and hour label format
            rangeslider=dict(visible=True)
            ),
        title=title,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
            )
        )

    if 'y1' in y_titles:
        fig.update_layout(yaxis=dict(title=y_titles['y1']))
    if 'y2' in y_titles:
        fig.update_layout(yaxis2=dict(
            title=y_titles['y2'],
            anchor="x",
            overlaying="y",
            side="right"
        ))
    if 'y3' in y_titles:
        fig.update_layout(yaxis3=dict(
            title=y_titles['y3'],
            anchor="free",
            overlaying="y",
            side="right",
            position=0.975
        ))

    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1d", step="day", stepmode="backward"),
                dict(count=7, label="1w", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )

    fig.update_yaxes(fixedrange=False)

    fig.update_layout(title_x=0.5) # center title

    return fig

def add_profile(fig, profile, yaxis='y', **kwargs) -> Figure:

    n_steps = len(profile)
    timestamps = pd.date_range(start='2000-01-01', periods=n_steps, freq='h')

    fig.add_trace(go.Scatter(
        x=timestamps,
        y=profile,
        connectgaps=False,
        yaxis=yaxis,
        **kwargs
        )
    )

    return fig