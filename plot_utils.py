from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.graph_objects as go
from centrex_tlf import utils

PeakHeightModel = Literal[
    "Thermal population",
    "Uniform hyperfine/mF population",
    "Equal peaks",
    "Lens mJ-selected",
    "Selected initial levels",
]
PEAK_HEIGHT_MODELS: tuple[PeakHeightModel, ...] = (
    "Thermal population",
    "Uniform hyperfine/mF population",
    "Equal peaks",
    "Lens mJ-selected",
    "Selected initial levels",
)
DEFAULT_POPULATION_J_VALUES = np.arange(13)


def gaussian(
    x: npt.NDArray[np.float64], mu: float, sig: float
) -> npt.NDArray[np.float64]:
    """Calculate Gaussian lineshape."""
    return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))


def _total_sublevels(j_values: npt.NDArray[np.integer]) -> float:
    return float(np.sum(utils.population.J_levels(j_values)))


def signal_amplitude(
    row: pd.Series,
    rotational_population: npt.NDArray[np.floating],
    height_model: PeakHeightModel = "Thermal population",
    population_j_values: npt.NDArray[np.integer] = DEFAULT_POPULATION_J_VALUES,
) -> float:
    if height_model == "Equal peaks":
        return 1.0

    strength = float(row["strength"])
    nphotons = float(row["nphotons"])
    population_idx = int(row["J_ground"])

    if height_model == "Uniform hyperfine/mF population":
        total_sublevels = _total_sublevels(population_j_values)
        return strength * nphotons / total_sublevels

    if height_model != "Thermal population":
        options = ", ".join(PEAK_HEIGHT_MODELS)
        raise ValueError(
            f"Unknown peak height model {height_model!r}; expected {options}."
        )

    if population_idx < 0 or population_idx >= len(rotational_population):
        population = 0.0
    else:
        population = rotational_population[population_idx]
    return (
        strength
        * nphotons
        * population
        / utils.population.J_levels(population_idx)
    )


def lens_mj_signal_amplitude(
    row: pd.Series,
    cluster_lines: pd.DataFrame,
    target_mj: int = 0,
    off_target_weight: float = 0.0,
) -> float:
    if "cluster_id" not in row:
        raise ValueError("Lens mJ-selected height model requires cluster_id.")
    required_columns = {"cluster_id", "ground_mJ_dominant", "strength", "nphotons"}
    missing_columns = required_columns.difference(cluster_lines.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Lens mJ-selected height model missing columns: {missing}.")

    cluster_id = int(row["cluster_id"])
    lines = cluster_lines[cluster_lines["cluster_id"].astype(int) == cluster_id]
    if lines.empty:
        return 0.0

    mjs = lines["ground_mJ_dominant"].to_numpy(dtype=int)
    weights = np.where(mjs == int(target_mj), 1.0, float(off_target_weight))
    return float(
        np.sum(
            lines["strength"].to_numpy(dtype=float)
            * lines["nphotons"].to_numpy(dtype=float)
            * weights
        )
    )


def selected_initial_level_signal_amplitude(
    row: pd.Series,
    cluster_lines: pd.DataFrame,
    selected_levels: set[tuple[int, float, int, int]] | None = None,
    selected_j: set[int] | None = None,
    selected_f1: set[float] | None = None,
    selected_f: set[int] | None = None,
    selected_mf: set[int] | None = None,
) -> float:
    if "cluster_id" not in row:
        raise ValueError("Selected initial levels height model requires cluster_id.")
    required_columns = {
        "cluster_id",
        "J_ground",
        "F1_ground",
        "F_ground",
        "mF_ground",
        "strength",
        "nphotons",
    }
    missing_columns = required_columns.difference(cluster_lines.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Selected initial levels height model missing columns: {missing}."
        )

    cluster_id = int(row["cluster_id"])
    lines = cluster_lines[cluster_lines["cluster_id"].astype(int) == cluster_id]
    if lines.empty:
        return 0.0

    selected_levels = set() if selected_levels is None else set(selected_levels)
    if selected_levels:
        line_level_tuples = list(
            zip(
                lines["J_ground"].to_numpy(dtype=int),
                lines["F1_ground"].to_numpy(dtype=float),
                lines["F_ground"].to_numpy(dtype=int),
                lines["mF_ground"].to_numpy(dtype=int),
            )
        )
        mask = np.array([level in selected_levels for level in line_level_tuples])
        selected_lines = lines.loc[mask]
        return float(
            np.sum(
                selected_lines["strength"].to_numpy(dtype=float)
                * selected_lines["nphotons"].to_numpy(dtype=float)
            )
        )

    selected_j = set() if selected_j is None else set(selected_j)
    selected_f1 = set() if selected_f1 is None else set(selected_f1)
    selected_f = set() if selected_f is None else set(selected_f)
    selected_mf = set() if selected_mf is None else set(selected_mf)

    mask = np.ones(len(lines), dtype=bool)
    if selected_j:
        mask &= np.isin(lines["J_ground"].to_numpy(dtype=int), list(selected_j))
    if selected_f1:
        mask &= np.isin(lines["F1_ground"].to_numpy(dtype=float), list(selected_f1))
    if selected_f:
        mask &= np.isin(lines["F_ground"].to_numpy(dtype=int), list(selected_f))
    if selected_mf:
        mask &= np.isin(lines["mF_ground"].to_numpy(dtype=int), list(selected_mf))

    selected_lines = lines.loc[mask]
    return float(
        np.sum(
            selected_lines["strength"].to_numpy(dtype=float)
            * selected_lines["nphotons"].to_numpy(dtype=float)
        )
    )


def generate_grid_plot(
    transition_name: str,
    visible_clusters: pd.DataFrame,
    rotational_population: npt.NDArray[np.floating],
    ir_uv: str = "IR",
    selected_cluster_ids: set[int] | None = None,
    height_model: PeakHeightModel = "Thermal population",
    linewidth_mhz: float = 2.5,
    cluster_lines: pd.DataFrame | None = None,
    lens_target_mj: int = 0,
    lens_off_target_weight: float = 0.0,
    selected_initial_levels: set[tuple[int, float, int, int]] | None = None,
    selected_initial_j: set[int] | None = None,
    selected_initial_f1: set[float] | None = None,
    selected_initial_f: set[int] | None = None,
    selected_initial_mf: set[int] | None = None,
) -> go.Figure:
    energy_axis = np.linspace(-15, 15, 201)
    lineshape = gaussian(energy_axis, 0, max(float(linewidth_mhz), np.finfo(float).eps))
    fig = go.Figure()
    delta_frequency_column = f"Δ freq [{ir_uv}, MHz]"

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
        if height_model == "Lens mJ-selected":
            if cluster_lines is None:
                raise ValueError(
                    "Lens mJ-selected height model requires cluster_lines."
                )
            amplitude = lens_mj_signal_amplitude(
                row,
                cluster_lines,
                target_mj=lens_target_mj,
                off_target_weight=lens_off_target_weight,
            )
        elif height_model == "Selected initial levels":
            if cluster_lines is None:
                raise ValueError(
                    "Selected initial levels height model requires cluster_lines."
                )
            amplitude = selected_initial_level_signal_amplitude(
                row,
                cluster_lines,
                selected_levels=selected_initial_levels,
                selected_j=selected_initial_j,
                selected_f1=selected_initial_f1,
                selected_f=selected_initial_f,
                selected_mf=selected_initial_mf,
            )
        else:
            amplitude = signal_amplitude(
                row,
                rotational_population,
                height_model,
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
                y=lineshape * amplitude,
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
