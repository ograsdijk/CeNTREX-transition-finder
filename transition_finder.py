from pathlib import Path

import numpy as np
import numpy.typing as npt
import streamlit as st
from centrex_tlf import utils

from calibration import R0_F1_1_2_F_1
from plot_utils import generate_grid_plot
from transition_grid import (
    GRID_ARTIFACT,
    TransitionGrid,
    cluster_transition_components,
    enrich_cluster_display_fields,
    format_cluster_dataframe,
    load_transition_grid,
    select_clusters_for_display,
)

st.set_page_config(page_title="CeNTREX Transitions", layout="wide")

file_path = Path(__file__).parent.absolute()

IR_UV_CONVERSION_FACTOR = 4
DEFAULT_ENERGY_RANGE = 300
DEFAULT_RESOLVING_FREQUENCY = 2.5


def initialize_session_state() -> None:
    defaults = {
        "energy_min_val": -DEFAULT_ENERGY_RANGE,
        "energy_max_val": DEFAULT_ENERGY_RANGE,
        "prev_ir_uv": "IR",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def adjust_energy_range_for_mode(new_mode: str) -> None:
    if st.session_state["prev_ir_uv"] != new_mode:
        factor = (
            IR_UV_CONVERSION_FACTOR if new_mode == "UV" else 1 / IR_UV_CONVERSION_FACTOR
        )
        st.session_state["energy_min_val"] = int(
            st.session_state["energy_min_val"] * factor
        )
        st.session_state["energy_max_val"] = int(
            st.session_state["energy_max_val"] * factor
        )
        st.session_state["prev_ir_uv"] = new_mode


def calibration_offset_from_grid(grid: TransitionGrid, cesium_frequency_ghz: float) -> float:
    zero_slice = grid.closest_slice(0.0)
    calibration_transition = R0_F1_1_2_F_1
    selected = zero_slice.lines[
        zero_slice.lines["transition_name"] == calibration_transition.transition.name
    ]
    if selected.empty:
        raise ValueError(
            f"Transition {calibration_transition.transition.name!r} not found in zero-field slice."
        )

    weights = selected["strength"].to_numpy(dtype=float)
    frequencies = selected["frequency_ir_mhz"].to_numpy(dtype=float)
    if np.sum(weights) > 0:
        model_frequency = float(np.average(frequencies, weights=weights))
    else:
        model_frequency = float(np.mean(frequencies))

    offset = -model_frequency + calibration_transition.frequency
    offset -= calibration_transition.cesium_frequency - cesium_frequency_ghz * 1e3
    return offset


def render_grid_app() -> None:
    if "transition_grid" not in st.session_state:
        st.session_state["transition_grid"] = load_transition_grid(
            file_path / GRID_ARTIFACT
        )
    grid = st.session_state["transition_grid"]
    metadata = getattr(grid, "metadata", {})
    if (
        metadata.get("schema_version", 0) < 3
        or metadata.get("frequency_units") != "IR MHz"
    ):
        st.error(
            "transition_grid.pkl was generated with an old frequency convention. "
            "Rerun 'uv run python compute_transitions.py' to rebuild it."
        )
        return

    with st.sidebar:
        st.title("Transition finder")
        ez_values = [float(v) for v in grid.ez_values]
        ez_v_cm = st.selectbox(
            label="Electric Field Ez [V/cm]", options=ez_values, index=0
        )
        resolving_frequency = st.number_input(
            label="Resolving Frequency [IR, MHz]",
            value=DEFAULT_RESOLVING_FREQUENCY,
            min_value=0.0,
            step=0.5,
            format="%.1f",
        )
        grid_slice = grid.closest_slice(float(ez_v_cm))
        clusters, cluster_lines = cluster_transition_components(
            grid_slice.lines,
            resolving_frequency_mhz=float(resolving_frequency),
            include_cluster_lines=True,
        )
        transition_names = sorted(
            str(name) for name in clusters["transition_name"].unique().tolist()
        )
        if not transition_names:
            st.error("No clustered transitions are available for this field value.")
            return
        transition_selector = st.selectbox(label="Transition", options=transition_names)
        transition_types = st.multiselect(
            label="Transition Types",
            options=["P", "Q", "R"],
            default=["P", "Q", "R"],
        )
        ir_uv = st.selectbox(label="UV or IR", options=["IR", "UV"], index=0)
        adjust_energy_range_for_mode(ir_uv)
        col1, col2 = st.columns(2)
        with col1:
            st.number_input(label="MHz", max_value=0, step=1, key="energy_min_val")
        with col2:
            st.number_input(label="MHz", min_value=0, step=1, key="energy_max_val")
        calibration_transition = R0_F1_1_2_F_1
        cesium_frequency = st.number_input(
            label="Cesium Frequency [GHz]",
            value=calibration_transition.cesium_frequency / 1e3,
            step=1e-3,
            format="%.3f",
        )
        rotational_temperature = st.number_input(
            label="Rotational Temperature [K]",
            value=6.5,
            step=0.1,
            format="%.1f",
        )

    if not transition_types:
        st.error("Select at least one transition type.")
        return
    clusters = clusters[clusters["branch"].isin(transition_types)].reset_index(
        drop=True
    )
    if transition_selector[0] not in transition_types:
        st.error(
            f"Selected transition type '{transition_selector[0]}' is not in the filter."
        )
        return

    energy_lim = (
        st.session_state["energy_min_val"],
        st.session_state["energy_max_val"],
    )
    thermal_population: npt.NDArray[np.floating] = utils.population.thermal_population(
        np.arange(13), T=rotational_temperature
    )
    calibration = calibration_offset_from_grid(grid, cesium_frequency)
    visible_clusters = select_clusters_for_display(
        clusters,
        transition_selector,
        energy_lim,
        ir_uv,
        calibration_offset_ir_mhz=calibration,
    )
    visible_clusters = enrich_cluster_display_fields(
        visible_clusters,
        cluster_lines,
        include_full_detail=True,
    )
    df = format_cluster_dataframe(visible_clusters, ir_uv)
    table_source_df = df.reset_index()
    summary_table_key = (
        f"cluster_summary::{float(ez_v_cm):g}::{float(resolving_frequency):g}::{ir_uv}::{transition_selector}"
    )
    table_state = st.session_state.get(summary_table_key, {})
    selected_rows = table_state.get("selection", {}).get("rows", [])
    selected_cluster_ids: set[int] = set()
    for selected_row in selected_rows:
        row_index = int(selected_row)
        if 0 <= row_index < len(table_source_df):
            selected_cluster_ids.add(int(table_source_df.iloc[row_index]["cluster_id"]))
    if not selected_cluster_ids:
        selected_cluster_ids = set(
            table_source_df.loc[
                table_source_df["transition"] == transition_selector,
                "cluster_id",
            ].astype(int)
        )

    fig = generate_grid_plot(
        transition_selector,
        visible_clusters,
        thermal_population,
        ir_uv,
        selected_cluster_ids=selected_cluster_ids,
    )
    st.plotly_chart(fig, width="stretch")

    summary_columns = [
        column
        for column in [
            f"delta frequency [{ir_uv}, MHz]",
            f"frequency [{ir_uv}, GHz]",
            "P'",
            "ΔmF",
            "dominant mF",
            "strength",
        ]
        if column in df.columns
    ]
    summary_df = table_source_df[["transition", *summary_columns]].rename(
        columns={"P'": "parity"}
    )
    if f"delta frequency [{ir_uv}, MHz]" in summary_df.columns:
        summary_df[f"delta frequency [{ir_uv}, MHz]"] = summary_df[
            f"delta frequency [{ir_uv}, MHz]"
        ].map(lambda value: f"{value:.1f}")
    if f"frequency [{ir_uv}, GHz]" in summary_df.columns:
        summary_df[f"frequency [{ir_uv}, GHz]"] = summary_df[
            f"frequency [{ir_uv}, GHz]"
        ].map(lambda value: f"{value:.3f}")
    if "strength" in summary_df.columns:
        summary_df["strength"] = summary_df["strength"].map(lambda value: f"{value:.3e}")

    st.dataframe(
        summary_df,
        hide_index=True,
        width="stretch",
        key=summary_table_key,
        on_select="rerun",
        selection_mode="multi-row",
    )
    with st.expander("Show full parent and mF detail"):
        detail_columns = [
            column
            for column in [
                "cluster_id",
                f"delta frequency [{ir_uv}, MHz]",
                f"frequency [{ir_uv}, GHz]",
                "P'",
                "spread [IR, MHz]",
                "state pairs",
                "mF branches",
                "photons",
                "excited parent",
                "mF detail",
            ]
            if column in df.columns
        ]
        detail_df = df.reset_index()[["transition", *detail_columns]].rename(
            columns={
                "P'": "parity",
                "spread [IR, MHz]": "span [IR, MHz]",
                "photons": "est. photons",
            }
        )
        if "cluster_id" in detail_df.columns:
            detail_df = detail_df.drop(columns=["cluster_id"])
        if f"delta frequency [{ir_uv}, MHz]" in detail_df.columns:
            detail_df[f"delta frequency [{ir_uv}, MHz]"] = detail_df[
                f"delta frequency [{ir_uv}, MHz]"
            ].map(lambda value: f"{value:.1f}")
        if f"frequency [{ir_uv}, GHz]" in detail_df.columns:
            detail_df[f"frequency [{ir_uv}, GHz]"] = detail_df[
                f"frequency [{ir_uv}, GHz]"
            ].map(lambda value: f"{value:.3f}")
        if "span [IR, MHz]" in detail_df.columns:
            detail_df["span [IR, MHz]"] = detail_df["span [IR, MHz]"].map(
                lambda value: f"{value:.2f}"
            )
        if "est. photons" in detail_df.columns:
            detail_df["est. photons"] = detail_df["est. photons"].map(
                lambda value: f"{value:.2f}"
            )
        st.dataframe(detail_df, hide_index=True, width="stretch")
    st.caption(
        f"Ez={grid_slice.ez_v_cm:g} V/cm, "
        f"{len(grid_slice.lines)} allowed E1 state pairs before clustering. "
        "Main table shows only the compact summary; expand below for span, pair counts, "
        "estimated photons, full parent labels, and full mF detail."
    )


initialize_session_state()
if not (file_path / GRID_ARTIFACT).exists():
    st.error(
        f"{GRID_ARTIFACT} not found. Run 'uv run python compute_transitions.py' "
        "to generate the required transition grid."
    )
else:
    render_grid_app()
