import sys
from pathlib import Path

import numpy as np
import pandas as pd
from centrex_tlf import utils

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plot_utils import (
    DEFAULT_POPULATION_J_VALUES,
    generate_grid_plot,
    lens_mj_signal_amplitude,
    marker_position_mhz,
    selected_initial_level_signal_amplitude,
    signal_amplitude,
)


def test_signal_amplitude_thermal_population_matches_existing_formula():
    row = pd.Series({"J_ground": 2, "strength": 3.0, "nphotons": 4.0})
    rotational_population = np.arange(13, dtype=float) / 10.0

    amplitude = signal_amplitude(
        row,
        rotational_population,
        height_model="Thermal population",
    )

    assert amplitude == (
        row["strength"]
        * row["nphotons"]
        * rotational_population[2]
        / utils.population.J_levels(2)
    )


def test_signal_amplitude_uniform_sublevel_population_is_j_independent():
    rotational_population = np.ones(13, dtype=float)
    total_sublevels = np.sum(utils.population.J_levels(DEFAULT_POPULATION_J_VALUES))
    row_j0 = pd.Series({"J_ground": 0, "strength": 3.0, "nphotons": 4.0})
    row_j5 = pd.Series({"J_ground": 5, "strength": 3.0, "nphotons": 4.0})

    amplitude_j0 = signal_amplitude(
        row_j0,
        rotational_population,
        height_model="Uniform hyperfine/mF population",
    )
    amplitude_j5 = signal_amplitude(
        row_j5,
        rotational_population,
        height_model="Uniform hyperfine/mF population",
    )

    assert amplitude_j0 == amplitude_j5
    assert amplitude_j0 == row_j0["strength"] * row_j0["nphotons"] / total_sublevels


def test_signal_amplitude_equal_peaks_ignores_weights():
    row = pd.Series({"J_ground": 7, "strength": 0.0, "nphotons": 0.0})
    rotational_population = np.zeros(13, dtype=float)

    amplitude = signal_amplitude(
        row,
        rotational_population,
        height_model="Equal peaks",
    )

    assert amplitude == 1.0


def test_lens_mj_signal_amplitude_uses_only_target_mj_by_default():
    row = pd.Series({"cluster_id": 10})
    cluster_lines = pd.DataFrame(
        [
            {
                "cluster_id": 10,
                "ground_mJ_dominant": 0,
                "strength": 2.0,
                "nphotons": 3.0,
            },
            {
                "cluster_id": 10,
                "ground_mJ_dominant": 1,
                "strength": 5.0,
                "nphotons": 7.0,
            },
        ]
    )

    amplitude = lens_mj_signal_amplitude(row, cluster_lines, target_mj=0)

    assert amplitude == 6.0


def test_lens_mj_signal_amplitude_applies_off_target_weight():
    row = pd.Series({"cluster_id": 10})
    cluster_lines = pd.DataFrame(
        [
            {
                "cluster_id": 10,
                "ground_mJ_dominant": 0,
                "strength": 2.0,
                "nphotons": 3.0,
            },
            {
                "cluster_id": 10,
                "ground_mJ_dominant": 1,
                "strength": 5.0,
                "nphotons": 7.0,
            },
        ]
    )

    amplitude = lens_mj_signal_amplitude(
        row,
        cluster_lines,
        target_mj=0,
        off_target_weight=0.1,
    )

    assert amplitude == 9.5


def test_selected_initial_level_signal_amplitude_uses_matching_component_lines():
    row = pd.Series({"cluster_id": 10})
    cluster_lines = pd.DataFrame(
        [
            {
                "cluster_id": 10,
                "J_ground": 2,
                "F1_ground": 3.5,
                "F_ground": 3,
                "mF_ground": 0,
                "strength": 2.0,
                "nphotons": 3.0,
            },
            {
                "cluster_id": 10,
                "J_ground": 2,
                "F1_ground": 3.5,
                "F_ground": 4,
                "mF_ground": 1,
                "strength": 5.0,
                "nphotons": 7.0,
            },
        ]
    )

    amplitude = selected_initial_level_signal_amplitude(
        row,
        cluster_lines,
        selected_j={2},
        selected_f1={3.5},
        selected_f={3},
        selected_mf={0},
    )

    assert amplitude == 6.0


def test_selected_initial_level_signal_amplitude_allows_multiple_values():
    row = pd.Series({"cluster_id": 10})
    cluster_lines = pd.DataFrame(
        [
            {
                "cluster_id": 10,
                "J_ground": 1,
                "F1_ground": 1.5,
                "F_ground": 1,
                "mF_ground": -1,
                "strength": 2.0,
                "nphotons": 3.0,
            },
            {
                "cluster_id": 10,
                "J_ground": 2,
                "F1_ground": 3.5,
                "F_ground": 3,
                "mF_ground": 0,
                "strength": 5.0,
                "nphotons": 7.0,
            },
            {
                "cluster_id": 10,
                "J_ground": 3,
                "F1_ground": 4.5,
                "F_ground": 4,
                "mF_ground": 1,
                "strength": 11.0,
                "nphotons": 13.0,
            },
        ]
    )

    amplitude = selected_initial_level_signal_amplitude(
        row,
        cluster_lines,
        selected_j={1, 2},
        selected_f1={1.5, 3.5},
    )

    assert amplitude == 41.0


def test_selected_initial_level_signal_amplitude_empty_sets_mean_all():
    row = pd.Series({"cluster_id": 10})
    cluster_lines = pd.DataFrame(
        [
            {
                "cluster_id": 10,
                "J_ground": 1,
                "F1_ground": 1.5,
                "F_ground": 1,
                "mF_ground": -1,
                "strength": 2.0,
                "nphotons": 3.0,
            },
            {
                "cluster_id": 10,
                "J_ground": 2,
                "F1_ground": 3.5,
                "F_ground": 3,
                "mF_ground": 0,
                "strength": 5.0,
                "nphotons": 7.0,
            },
        ]
    )

    amplitude = selected_initial_level_signal_amplitude(
        row,
        cluster_lines,
        selected_j=set(),
        selected_f1=set(),
        selected_f=set(),
        selected_mf=set(),
    )

    assert amplitude == 41.0


def test_selected_initial_level_signal_amplitude_returns_zero_without_match():
    row = pd.Series({"cluster_id": 10})
    cluster_lines = pd.DataFrame(
        [
            {
                "cluster_id": 10,
                "J_ground": 2,
                "F1_ground": 3.5,
                "F_ground": 3,
                "mF_ground": 0,
                "strength": 2.0,
                "nphotons": 3.0,
            },
        ]
    )

    amplitude = selected_initial_level_signal_amplitude(
        row,
        cluster_lines,
        selected_j={1},
    )

    assert amplitude == 0.0


def test_selected_initial_level_signal_amplitude_matches_explicit_level_combinations():
    row = pd.Series({"cluster_id": 10})
    cluster_lines = pd.DataFrame(
        [
            {
                "cluster_id": 10,
                "J_ground": 1,
                "F1_ground": 1.5,
                "F_ground": 1,
                "mF_ground": -1,
                "strength": 2.0,
                "nphotons": 3.0,
            },
            {
                "cluster_id": 10,
                "J_ground": 1,
                "F1_ground": 1.5,
                "F_ground": 1,
                "mF_ground": 0,
                "strength": 5.0,
                "nphotons": 7.0,
            },
            {
                "cluster_id": 10,
                "J_ground": 1,
                "F1_ground": 1.5,
                "F_ground": 2,
                "mF_ground": -1,
                "strength": 11.0,
                "nphotons": 13.0,
            },
        ]
    )

    amplitude = selected_initial_level_signal_amplitude(
        row,
        cluster_lines,
        selected_levels={(1, 1.5, 1, -1)},
    )

    assert amplitude == 6.0


def test_selected_initial_level_signal_amplitude_prefers_explicit_levels_over_sets():
    row = pd.Series({"cluster_id": 10})
    cluster_lines = pd.DataFrame(
        [
            {
                "cluster_id": 10,
                "J_ground": 1,
                "F1_ground": 1.5,
                "F_ground": 1,
                "mF_ground": -1,
                "strength": 2.0,
                "nphotons": 3.0,
            },
            {
                "cluster_id": 10,
                "J_ground": 1,
                "F1_ground": 1.5,
                "F_ground": 2,
                "mF_ground": -1,
                "strength": 11.0,
                "nphotons": 13.0,
            },
        ]
    )

    amplitude = selected_initial_level_signal_amplitude(
        row,
        cluster_lines,
        selected_levels={(1, 1.5, 1, -1)},
        selected_j={1},
        selected_f1={1.5},
        selected_f={1, 2},
        selected_mf={-1},
    )

    assert amplitude == 6.0


def test_generate_grid_plot_uses_linewidth_for_gaussian_width():
    visible_clusters = pd.DataFrame(
        [
            {
                "cluster_id": 0,
                "transition_name": "R(2) F1'=7/2 F'=3",
                "branch": "R",
                "J_ground": 2,
                "strength": 1.0,
                "nphotons": 1.0,
                "Δ freq [IR, MHz]": 0.0,
            }
        ]
    )
    rotational_population = np.ones(13, dtype=float)

    narrow = generate_grid_plot(
        "R(2) F1'=7/2 F'=3",
        visible_clusters,
        rotational_population,
        height_model="Equal peaks",
        linewidth_mhz=1.0,
    )
    broad = generate_grid_plot(
        "R(2) F1'=7/2 F'=3",
        visible_clusters,
        rotational_population,
        height_model="Equal peaks",
        linewidth_mhz=4.0,
    )

    narrow_y = np.asarray(narrow.data[0].y, dtype=float)
    broad_y = np.asarray(broad.data[0].y, dtype=float)
    offset_idx = int(np.argmin(np.abs(np.asarray(narrow.data[0].x, dtype=float) - 4.0)))

    assert np.max(narrow_y) == np.max(broad_y)
    assert broad_y[offset_idx] > narrow_y[offset_idx]


def test_marker_position_converts_absolute_frequency_to_delta_axis():
    position = marker_position_mhz(
        0.484,
        "UV GHz",
        ir_uv="UV",
        reference_frequency_ir_mhz=100.0,
        calibration_offset_ir_mhz=1.0,
    )

    assert position == 80.0


def test_generate_grid_plot_renders_measured_marker_trace():
    delta = chr(916)
    visible_clusters = pd.DataFrame(
        [
            {
                "cluster_id": 0,
                "transition_name": "R(2) F1'=7/2 F'=3",
                "branch": "R",
                "J_ground": 2,
                "strength": 1.0,
                "nphotons": 1.0,
                f"{delta} freq [IR, MHz]": 0.0,
            }
        ]
    )
    rotational_population = np.ones(13, dtype=float)
    markers = pd.DataFrame(
        [
            {
                "label": "measured",
                "frequency": 125.0,
                "scale": "IR MHz",
                "color": "#ff0000",
                "note": "scan peak",
            }
        ]
    )

    fig = generate_grid_plot(
        "R(2) F1'=7/2 F'=3",
        visible_clusters,
        rotational_population,
        height_model="Equal peaks",
        measured_markers=markers,
        reference_frequency_ir_mhz=100.0,
        calibration_offset_ir_mhz=5.0,
    )

    marker_trace = fig.data[-4]
    assert marker_trace.name == "measured"
    assert list(marker_trace.x) == [20.0, 20.0]
    assert marker_trace.line.dash == "dash"


def test_generate_grid_plot_defaults_blank_marker_color():
    delta = chr(916)
    visible_clusters = pd.DataFrame(
        [
            {
                "cluster_id": 0,
                "transition_name": "R(2) F1'=7/2 F'=3",
                "branch": "R",
                "J_ground": 2,
                "strength": 1.0,
                "nphotons": 1.0,
                f"{delta} freq [IR, MHz]": 0.0,
            }
        ]
    )
    markers = pd.DataFrame(
        [
            {
                "label": "measured",
                "frequency": 20.0,
                "scale": "Δ MHz",
                "color": None,
                "note": "",
            }
        ]
    )

    fig = generate_grid_plot(
        "R(2) F1'=7/2 F'=3",
        visible_clusters,
        np.ones(13, dtype=float),
        height_model="Equal peaks",
        measured_markers=markers,
        reference_frequency_ir_mhz=100.0,
    )

    marker_trace = fig.data[-4]
    assert marker_trace.line.color == "#d62728"


def test_generate_grid_plot_renders_single_vertical_marker_trace():
    delta = chr(916)
    visible_clusters = pd.DataFrame(
        [
            {
                "cluster_id": 0,
                "transition_name": "R(2) F1'=7/2 F'=3",
                "branch": "R",
                "J_ground": 2,
                "strength": 1.0,
                "nphotons": 1.0,
                f"{delta} freq [IR, MHz]": 0.0,
            }
        ]
    )

    fig = generate_grid_plot(
        "R(2) F1'=7/2 F'=3",
        visible_clusters,
        np.ones(13, dtype=float),
        height_model="Equal peaks",
        vertical_marker_mhz=12.5,
        vertical_marker_label="marker 12.5",
    )

    marker_trace = fig.data[-4]
    assert marker_trace.name == "marker 12.5"
    assert list(marker_trace.x) == [12.5, 12.5]
    assert marker_trace.line.dash == "dash"


def test_generate_grid_plot_omits_single_vertical_marker_when_disabled():
    delta = chr(916)
    visible_clusters = pd.DataFrame(
        [
            {
                "cluster_id": 0,
                "transition_name": "R(2) F1'=7/2 F'=3",
                "branch": "R",
                "J_ground": 2,
                "strength": 1.0,
                "nphotons": 1.0,
                f"{delta} freq [IR, MHz]": 0.0,
            }
        ]
    )

    fig = generate_grid_plot(
        "R(2) F1'=7/2 F'=3",
        visible_clusters,
        np.ones(13, dtype=float),
        height_model="Equal peaks",
        vertical_marker_mhz=None,
    )

    assert all(trace.name != "marker" for trace in fig.data)


def test_generate_grid_plot_can_normalize_peak_height():
    delta = chr(916)
    visible_clusters = pd.DataFrame(
        [
            {
                "cluster_id": 0,
                "transition_name": "R(2) F1'=7/2 F'=3",
                "branch": "R",
                "J_ground": 2,
                "strength": 5.0,
                "nphotons": 1.0,
                f"{delta} freq [IR, MHz]": 0.0,
            }
        ]
    )

    fig = generate_grid_plot(
        "R(2) F1'=7/2 F'=3",
        visible_clusters,
        np.ones(13, dtype=float),
        height_model="Thermal population",
        normalize_heights=True,
    )

    assert np.max(np.asarray(fig.data[0].y, dtype=float)) == 1.0
