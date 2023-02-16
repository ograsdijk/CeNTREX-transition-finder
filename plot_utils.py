import numpy as np
import plotly.graph_objects as go
from centrex_tlf import transitions

from hamiltonian_utils import SortedTransitions


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))


def generate_plot(
    transitions_interest: list[transitions.OpticalTransition],
    sorted_transitions: SortedTransitions,
    energy_lim: tuple[float, float] = (-300, 300),
    ir_uv: str = "IR",
) -> go.Figure:
    if ir_uv == "IR":
        convert = 1
    else:
        convert = 4

    _energy = np.linspace(-15, 15, 201)
    lineshape = gaussian(_energy, 0, 2.5)

    fig = go.Figure()

    idx_center = np.where(sorted_transitions.transitions == transitions_interest)[0]
    offset = sorted_transitions.energies[idx_center]

    mask = ((sorted_transitions.energies - offset) * convert >= energy_lim[0]) & (
        (sorted_transitions.energies - offset) * convert <= energy_lim[1]
    )

    for trans, loc in zip(
        sorted_transitions.transitions[mask], sorted_transitions.energies[mask]
    ):
        if trans == transitions_interest:
            color = "black"
        elif trans.t == transitions.OpticalTransitionType.R:
            color = "#9467bd"
        elif trans.t == transitions.OpticalTransitionType.Q:
            color = "#ff7f0e"
        elif trans.t == transitions.OpticalTransitionType.P:
            color = "#2ca02c"
        fig.add_trace(
            go.Scatter(
                x=(_energy + loc - offset) * convert,
                y=lineshape,
                name=trans.name,
                line={"color": color},
                meta=[trans.name],
                hovertemplate="%{meta[0]}",
                mode="lines",
                showlegend=False,
            )
        )

    for trans, color in [("R", "#9467bd"), ("Q", "#ff7f0e"), ("P", "#2ca02c")]:
        fig.add_trace(
            go.Scatter(
                x=[np.nan], y=[np.nan], name=trans, mode="lines", line=dict(color=color)
            )
        )
    fig.update_layout(
        title="Frequency scan",
        xaxis_title=f"frequency [{ir_uv}, MHz]",
        # yaxis_title="fluorescence [arb]",
        legend_title="Transition type",
        font=dict(size=14)
        # font=dict(
        #     family="Courier New, monospace",
        #     size=18,
        #     color="RebeccaPurple"
        # )
    )
    fig.update_layout(showlegend=True)
    return fig
