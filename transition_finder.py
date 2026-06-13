from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import streamlit as st
from centrex_tlf import utils

from calibration import DEFAULT_CALIBRATION_TRANSITION
from plot_utils import PEAK_HEIGHT_MODELS, generate_grid_plot
from transition_grid import (
    GRID_ARTIFACT,
    POLARIZATION_STRENGTH_COLUMNS,
    TransitionGrid,
    apply_polarization_selection,
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
INITIAL_LEVEL_COLUMNS = ("J_ground", "F1_ground", "F_ground", "mF_ground")


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


def calibration_offset_from_grid(
    grid: TransitionGrid, cesium_frequency_ghz: float
) -> float:
    zero_slice = grid.closest_slice(0.0)
    calibration_transition = DEFAULT_CALIBRATION_TRANSITION
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


def _format_half_integer(value: float) -> str:
    numerator = round(float(value) * 2)
    if numerator % 2 == 0:
        return str(numerator // 2)
    return f"{numerator}/2"


def _visible_cluster_lines(
    cluster_lines: pd.DataFrame,
    visible_cluster_ids: set[int],
) -> pd.DataFrame:
    if cluster_lines.empty or not visible_cluster_ids:
        return cluster_lines.iloc[0:0]
    return cluster_lines[
        cluster_lines["cluster_id"].astype(int).isin(visible_cluster_ids)
    ]


def _sorted_initial_level_combinations(
    frame: pd.DataFrame,
) -> list[tuple[int, float, int, int]]:
    required = ["J_ground", "F1_ground", "F_ground", "mF_ground"]
    if not set(required).issubset(frame.columns):
        return []
    unique_levels = frame[required].dropna().drop_duplicates()
    combinations = [
        (int(row["J_ground"]), float(row["F1_ground"]), int(row["F_ground"]), int(row["mF_ground"]))
        for _, row in unique_levels.iterrows()
    ]
    return sorted(combinations)


def _format_initial_level_combination(level: tuple[int, float, int, int]) -> str:
    j_ground, f1_ground, f_ground, mf_ground = level
    return (
        f"J={j_ground}, F1={_format_half_integer(f1_ground)}, "
        f"F={f_ground}, mF={mf_ground}"
    )


def render_grid_app() -> None:
    if "transition_grid" not in st.session_state:
        st.session_state["transition_grid"] = load_transition_grid(
            file_path / GRID_ARTIFACT
        )
    grid = st.session_state["transition_grid"]
    metadata = getattr(grid, "metadata", {})
    if (
        metadata.get("schema_version", 0) < 4
        or metadata.get("frequency_units") != "IR MHz"
    ):
        st.error(
            "transition_grid.pkl was generated with an old grid schema. "
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
        polarization = st.selectbox(
            label="Polarization",
            options=list(POLARIZATION_STRENGTH_COLUMNS),
            index=0,
        )
        grid_slice = grid.closest_slice(float(ez_v_cm))
        selected_lines = apply_polarization_selection(grid_slice.lines, polarization)
        if selected_lines.empty:
            st.error(f"No allowed E1 state pairs remain for {polarization} polarization.")
            return
        has_mj_population_data = "ground_mJ_dominant" in selected_lines.columns
        has_initial_level_population_data = set(INITIAL_LEVEL_COLUMNS).issubset(
            selected_lines.columns
        )
        clusters, cluster_lines = cluster_transition_components(
            selected_lines,
            resolving_frequency_mhz=float(resolving_frequency),
            include_cluster_lines=True,
        )
        zero_field_clusters = None
        if not np.isclose(grid_slice.ez_v_cm, 0.0):
            zero_field_lines = apply_polarization_selection(
                grid.closest_slice(0.0).lines, polarization
            )
            zero_field_clusters = cluster_transition_components(
                zero_field_lines,
                resolving_frequency_mhz=float(resolving_frequency),
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
        calibration_transition = DEFAULT_CALIBRATION_TRANSITION
        cesium_frequency = st.number_input(
            label="Cesium Frequency [GHz]",
            value=calibration_transition.cesium_frequency / 1e3,
            step=1e-3,
            format="%.3f",
        )

    if not transition_types:
        st.error("Select at least one transition type.")
        return
    clusters = clusters[clusters["branch"].isin(transition_types)].reset_index(
        drop=True
    )
    if zero_field_clusters is not None:
        zero_field_clusters = zero_field_clusters[
            zero_field_clusters["branch"].isin(transition_types)
        ].reset_index(drop=True)
    if transition_selector[0] not in transition_types:
        st.error(
            f"Selected transition type '{transition_selector[0]}' is not in the filter."
        )
        return

    energy_lim = (
        st.session_state["energy_min_val"],
        st.session_state["energy_max_val"],
    )
    calibration = calibration_offset_from_grid(grid, cesium_frequency)
    visible_clusters = select_clusters_for_display(
        clusters,
        transition_selector,
        energy_lim,
        ir_uv,
        calibration_offset_ir_mhz=calibration,
        zero_field_clusters=zero_field_clusters,
    )
    visible_cluster_ids = set(visible_clusters["cluster_id"].astype(int))
    visible_clusters = enrich_cluster_display_fields(
        visible_clusters,
        cluster_lines,
        include_full_detail=True,
    )
    visible_lines = _visible_cluster_lines(cluster_lines, visible_cluster_ids)

    with st.sidebar.container(border=True):
        st.subheader("Population")
        population_models = list(PEAK_HEIGHT_MODELS)
        if not has_mj_population_data:
            population_models = [
                model for model in population_models if model != "Lens mJ-selected"
            ]
        if not has_initial_level_population_data:
            population_models = [
                model
                for model in population_models
                if model != "Selected initial levels"
            ]
        height_model = st.selectbox(
            label="Population mode",
            options=population_models,
            index=(
                population_models.index("Equal peaks")
                if "Equal peaks" in population_models
                else 0
            ),
            label_visibility="collapsed",
        )
        if not has_mj_population_data:
            st.caption(f"Rebuild {GRID_ARTIFACT} to enable mJ population weighting.")
        if not has_initial_level_population_data:
            st.caption(
                f"Rebuild {GRID_ARTIFACT} to enable initial-level population selection."
            )

        rotational_temperature = 6.5
        lens_target_mj = 0
        lens_off_target_weight = 0.0
        selected_initial_levels: list[tuple[int, float, int, int]] = []
        if height_model == "Thermal population":
            rotational_temperature = st.number_input(
                label="Rotational Temperature [K]",
                value=6.5,
                step=0.1,
                format="%.1f",
            )
        elif height_model == "Lens mJ-selected":
            lens_target_mj = st.number_input(
                label="Target mJ",
                value=0,
                step=1,
                format="%d",
            )
            lens_off_target_weight = st.number_input(
                label="Off-target weight",
                value=0.0,
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                format="%.2f",
            )
        elif height_model == "Selected initial levels":
            initial_level_options = _sorted_initial_level_combinations(visible_lines)
            selected_initial_levels = st.multiselect(
                label="Initial levels (J, F1, F, mF)",
                options=initial_level_options,
                default=[],
                format_func=_format_initial_level_combination,
            )
            st.caption(f"Selected combinations: {len(selected_initial_levels)}")
            if selected_initial_levels:
                selected_levels_preview = pd.DataFrame(
                    [
                        {
                            "J": int(level[0]),
                            "F1": _format_half_integer(level[1]),
                            "F": int(level[2]),
                            "mF": int(level[3]),
                        }
                        for level in selected_initial_levels
                    ]
                )
                st.dataframe(
                    selected_levels_preview,
                    hide_index=True,
                    width="stretch",
                    height=min(240, 35 * len(selected_levels_preview) + 36),
                )

    if height_model == "Thermal population":
        rotational_population: npt.NDArray[np.floating] = (
            utils.population.thermal_population(np.arange(13), T=rotational_temperature)
        )
    else:
        rotational_population = np.ones(13, dtype=float)

    df = format_cluster_dataframe(visible_clusters, ir_uv)
    table_source_df = df.reset_index()
    summary_table_key = (
        f"cluster_summary::{float(ez_v_cm):g}::{float(resolving_frequency):g}::"
        f"{polarization}::{ir_uv}::{transition_selector}"
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
        rotational_population,
        ir_uv,
        selected_cluster_ids=selected_cluster_ids,
        height_model=height_model,
        linewidth_mhz=float(resolving_frequency),
        cluster_lines=cluster_lines,
        lens_target_mj=int(lens_target_mj) if height_model == "Lens mJ-selected" else 0,
        lens_off_target_weight=(
            float(lens_off_target_weight)
            if height_model == "Lens mJ-selected"
            else 0.0
        ),
        selected_initial_levels=set(selected_initial_levels)
        if height_model == "Selected initial levels"
        else None,
        selected_initial_j=None,
        selected_initial_f1=None,
        selected_initial_f=None,
        selected_initial_mf=None,
    )
    st.plotly_chart(fig, width="stretch")

    summary_columns = [
        column
        for column in [
            f"Δ freq [{ir_uv}, MHz]",
            f"Δ from 0 V/cm [{ir_uv}, MHz]",
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
    if f"Δ freq [{ir_uv}, MHz]" in summary_df.columns:
        summary_df[f"Δ freq [{ir_uv}, MHz]"] = summary_df[
            f"Δ freq [{ir_uv}, MHz]"
        ].map(lambda value: f"{value:.1f}")
    if f"Δ from 0 V/cm [{ir_uv}, MHz]" in summary_df.columns:
        summary_df[f"Δ from 0 V/cm [{ir_uv}, MHz]"] = summary_df[
            f"Δ from 0 V/cm [{ir_uv}, MHz]"
        ].map(lambda value: f"{value:.1f}")
    if f"frequency [{ir_uv}, GHz]" in summary_df.columns:
        summary_df[f"frequency [{ir_uv}, GHz]"] = summary_df[
            f"frequency [{ir_uv}, GHz]"
        ].map(lambda value: f"{value:.3f}")
    if "strength" in summary_df.columns:
        summary_df["strength"] = summary_df["strength"].map(
            lambda value: f"{value:.3e}"
        )

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
                f"Δ freq [{ir_uv}, MHz]",
                f"Δ from 0 V/cm [{ir_uv}, MHz]",
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
        if f"Δ freq [{ir_uv}, MHz]" in detail_df.columns:
            detail_df[f"Δ freq [{ir_uv}, MHz]"] = detail_df[
                f"Δ freq [{ir_uv}, MHz]"
            ].map(lambda value: f"{value:.1f}")
        if f"Δ from 0 V/cm [{ir_uv}, MHz]" in detail_df.columns:
            detail_df[f"Δ from 0 V/cm [{ir_uv}, MHz]"] = detail_df[
                f"Δ from 0 V/cm [{ir_uv}, MHz]"
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
        f"{len(selected_lines)} {polarization} E1 state pairs before clustering "
        f"({len(grid_slice.lines)} in All). "
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
