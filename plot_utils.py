import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.graph_objects as go
from centrex_tlf import utils


def gaussian(
    x: npt.NDArray[np.float64], mu: float, sig: float
) -> npt.NDArray[np.float64]:
    """Calculate Gaussian lineshape."""
    return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))


def generate_grid_plot(
    transition_name: str,
    visible_clusters: pd.DataFrame,
    thermal_population: npt.NDArray[np.floating],
    ir_uv: str = "IR",
    selected_cluster_ids: set[int] | None = None,
) -> go.Figure:
    energy_axis = np.linspace(-15, 15, 201)
    lineshape = gaussian(energy_axis, 0, 2.5)
    fig = go.Figure()
    delta_frequency_column = f"delta frequency [{ir_uv}, MHz]"

    colors = {
        "P": "#2ca02c",
        "Q": "#ff7f0e",
        "R": "#1f77b4",
    }
    selected_cluster_ids = (
        set() if selected_cluster_ids is None else selected_cluster_ids
    )
    has_selection = len(selected_cluster_ids) > 0

    for _, row in visible_clusters.iterrows():
        is_selected = int(row["cluster_id"]) in selected_cluster_ids
        color = colors.get(row["branch"], "grey")
        population_idx = int(row["J_ground"])
        if population_idx >= len(thermal_population):
            population = 0.0
        else:
            population = thermal_population[population_idx]
        signal_amplitude = (
            row["strength"]
            * row["nphotons"]
            * population
            / utils.population.J_levels(population_idx)
        )
        label_parts = [str(row["transition_name"])]
        parent_parity = row.get("excited_parent_parity", "")
        if isinstance(parent_parity, str) and parent_parity and parent_parity != "?":
            label_parts.append(f"P={parent_parity}")
        mf_summary = row.get("mf_dominant", row.get("mF", ""))
        if isinstance(mf_summary, str) and mf_summary:
            label_parts.append(mf_summary)
        label = " | ".join(label_parts)
        fig.add_trace(
            go.Scatter(
                x=energy_axis + float(row[delta_frequency_column]),
                y=lineshape * signal_amplitude,
                name=label,
                line={"color": color, "width": 4 if is_selected else 2},
                meta=[label],
                hovertemplate="%{meta[0]}",
                mode="lines",
                opacity=1.0 if is_selected or not has_selection else 0.25,
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
