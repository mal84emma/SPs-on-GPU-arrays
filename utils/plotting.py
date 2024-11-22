"""Plotting utilities."""

import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.graph_objects import Figure
from plotly.subplots import make_subplots


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
            side="right",
            griddash="dash"
        ))
    if 'y3' in y_titles:
        fig.update_layout(yaxis3=dict(
            title=y_titles['y3'],
            anchor="free",
            overlaying="y",
            side="right",
            position=0.975,
            griddash="dot"
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
    timestamps = pd.date_range(start='2000-01-01', periods=n_steps, freq='H')

    fig.add_trace(go.Scatter(
        x=timestamps,
        y=profile,
        connectgaps=False,
        yaxis=yaxis,
        **kwargs
        )
    )

    return fig