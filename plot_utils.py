from typing import Optional, Sequence

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from centrex_tlf import transitions, utils

from generate_transitions import Transition


def gaussian(
    x: npt.NDArray[np.float64], mu: float, sig: float
) -> npt.NDArray[np.float64]:
    """Calculate Gaussian lineshape."""
    return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))


def generate_plot(
    transition_interest: transitions.OpticalTransition,
    sorted_transitions: Sequence[Transition],
    thermal_population: npt.NDArray[np.floating],
    energy_lim: tuple[float, float] = (-300, 300),
    ir_uv: str = "IR",
    transition_types: Optional[list[str]] = None,
) -> go.Figure:
    if ir_uv == "IR":
        convert = 1
    else:
        convert = 4

    _energy = np.linspace(-15, 15, 201)
    lineshape = gaussian(_energy, 0, 2.5)

    fig = go.Figure()

    idx_center = next(
        (
            idx
            for idx, obj in enumerate(sorted_transitions)
            if obj.transition == transition_interest
        ),
        None,  # default if not found
    )
    if idx_center is None:
        raise ValueError("Transition of interest not found in transitions list.")

    offset = sorted_transitions[idx_center].weighted_energy
    energies = np.array([trans.weighted_energy for trans in sorted_transitions])

    indices_select = np.nonzero(
        ((energies - offset) * convert >= energy_lim[0])
        & ((energies - offset) * convert <= energy_lim[1])
    )[0]

    colors = {
        "O": "#9467bd",
        "P": "#2ca02c",
        "Q": "#ff7f0e",
        "R": "#1f77b4",
        "S": "#d62728",
    }

    selected_transitions: list[Transition] = [
        sorted_transitions[i] for i in indices_select
    ]

    for trans in selected_transitions:
        # Filter by transition type if specified
        if transition_types and trans.transition.name[0] not in transition_types:
            continue

        energy = trans.weighted_energy
        amplitude = (np.abs(trans.coupling_elements_E1) ** 2).mean()

        if trans == transition_interest:
            color = "black"
        else:
            color = colors.get(trans.transition.name[0], "grey")

        ground_states_participating = trans.ground_states_participating

        signal_amplitude = (
            amplitude
            * trans.nphotons
            * thermal_population[trans.transition.J_ground]
            * ground_states_participating
            / utils.population.J_levels(trans.transition.J_ground)
        )

        fig.add_trace(
            go.Scatter(
                x=(_energy + energy - offset) * convert,
                y=lineshape * signal_amplitude,
                name=trans.transition.name,
                line={"color": color},
                meta=[trans.transition.name],
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
        legend_title="Transition type",
        font=dict(size=14),
        showlegend=True,
    )
    return fig
