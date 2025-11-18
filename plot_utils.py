import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from centrex_tlf import transitions, utils

from hamiltonian_utils import Transitions, unique_unsorted


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))


def generate_plot(
    transition_interest: transitions.OpticalTransition,
    sorted_transitions: Transitions,
    energy_lim: tuple[float, float] = (-300, 300),
    ir_uv: str = "IR",
    thermal_population: npt.NDArray[np.float64] = utils.population.thermal_population(
        np.arange(10), T=6.5
    ),
) -> go.Figure:
    if ir_uv == "IR":
        convert = 1
    else:
        convert = 4

    _energy = np.linspace(-15, 15, 201)
    lineshape = gaussian(_energy, 0, 2.5)

    fig = go.Figure()

    indices_center = np.where(sorted_transitions.transitions == transition_interest)[0]

    offset = sorted_transitions.energies_mean[indices_center]

    mask = ((sorted_transitions.energies_mean - offset) * convert >= energy_lim[0]) & (
        (sorted_transitions.energies_mean - offset) * convert <= energy_lim[1]
    )

    colors = {
        "O": "#9467bd",
        "P": "#2ca02c",
        "Q": "#ff7f0e",
        "R": "#1f77b4",
        "S": "#d62728",
    }

    for trans, trans_data in zip(
        sorted_transitions.transitions[mask], sorted_transitions.transitions_data[mask]
    ):
        coupling_elements = trans_data.coupling_elements
        mask_nonzero = coupling_elements != 0
        energy = trans_data.energies[mask_nonzero].mean()
        amplitude = np.abs(coupling_elements).mean()

        if trans == transition_interest:
            color = "black"
        else:
            color = colors.get(trans.t.name, "grey")

        signal_amplitude = (
            amplitude
            * trans_data.photons
            * thermal_population[trans.J_ground]
            / utils.population.J_levels(trans.J_ground)
            * len(unique_unsorted(trans_data.states_ground[mask_nonzero]))
        )

        fig.add_trace(
            go.Scatter(
                x=(_energy + energy - offset) * convert,
                y=lineshape * signal_amplitude,
                name=trans.name,
                line={"color": color},
                meta=[trans.name],
                hovertemplate="%{meta[0]}",
                mode="lines",
                showlegend=False,
            )
        )

    for trans_type, color in colors.items():
        fig.add_trace(
            go.Scatter(
                x=[np.nan],
                y=[np.nan],
                name=trans_type,
                mode="lines",
                line=dict(color=color),
            )
        )
    fig.update_layout(
        title="Frequency scan",
        xaxis_title=f"frequency [{ir_uv}, MHz]",
        # yaxis_title="fluorescence [arb]",
        legend_title="Transition type",
        font=dict(size=14),
        # font=dict(
        #     family="Courier New, monospace",
        #     size=18,
        #     color="RebeccaPurple"
        # )
    )
    fig.update_layout(showlegend=True)
    return fig
