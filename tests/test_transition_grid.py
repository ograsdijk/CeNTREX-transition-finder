import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from centrex_tlf import hamiltonian
from eigenshuffle import eigenshuffle_eigh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transition_grid import (
    E1_POLARIZATION_VECTORS,
    ElapsedSpinner,
    POLARIZATION_STRENGTH_COLUMNS,
    _assign_labels_by_overlap,
    _build_e1_operator_matrices,
    _build_hamiltonian_sequences,
    _dominant_uncoupled_mj_values,
    _polarization_strength_matrices,
    _tracking_boundary_permutation,
    _build_tracked_manifold,
    _build_transition_candidates,
    _coupling_matrices_for_field,
    _line_records_for_slice,
    apply_polarization_selection,
    build_arg_parser,
    build_transition_grid,
    cluster_transition_components,
    enrich_cluster_display_fields,
    format_elapsed,
    generate_cluster_dataframe,
    generate_ED_ME_mixed_state,
    transition_centroid,
)


def test_format_elapsed_uses_hh_mm_ss():
    assert format_elapsed(0.2) == "00:00:00"
    assert format_elapsed(65.9) == "00:01:05"
    assert format_elapsed(3661.0) == "01:01:01"
    assert format_elapsed(-5.0) == "00:00:00"


def test_spinner_disables_for_non_tty_stream():
    stream = StringIO()
    spinner = ElapsedSpinner("Building", stream=stream)

    assert not spinner.enabled


def test_arg_parser_accepts_no_spinner_flag():
    args = build_arg_parser().parse_args(
        [
            "--no-spinner",
            "--workers",
            "2",
            "--e1-workers",
            "4",
            "--tracking-workers",
            "3",
        ]
    )

    assert args.no_spinner
    assert args.workers == 2
    assert args.e1_workers == 4
    assert args.tracking_workers == 3


def test_build_transition_grid_uses_progress_reporter_callback():
    progress_lines: list[str] = []

    build_transition_grid(
        ez_values=[0.0],
        ground_js=[0],
        excited_js=[1],
        progress=True,
        progress_reporter=progress_lines.append,
    )

    assert len(progress_lines) == 2
    assert progress_lines[0].startswith("Setup:")
    assert "tracking=" in progress_lines[0]
    assert "(workers=1)" in progress_lines[0]
    assert progress_lines[1].startswith("Ez=0")


def test_build_transition_grid_eta_excludes_setup_time():
    class FakeTimer:
        def __init__(self, values: list[float]) -> None:
            self.values = values
            self.index = 0

        def __call__(self) -> float:
            if self.index >= len(self.values):
                return self.values[-1]
            value = self.values[self.index]
            self.index += 1
            return value

    timer = FakeTimer(
        [
            0.0,
            100.0,
            110.0,
            120.0,
            120.0,
            121.0,
            122.0,
            123.0,
            123.0,
            123.0,
            123.0,
            124.0,
            125.0,
            126.0,
        ]
    )
    progress_lines: list[str] = []

    build_transition_grid(
        ez_values=[0.0, 25.0],
        ground_js=[0],
        excited_js=[1],
        progress=True,
        progress_reporter=progress_lines.append,
        timer=timer,
    )

    assert progress_lines[0].startswith("Setup:")
    assert "elapsed=126.0s" in progress_lines[1]
    assert "eta=3.0s" in progress_lines[1]
    assert "eta=0.0s" in progress_lines[2]


def test_e1_operator_matrices_parallel_matches_serial():
    (
        _x_uncoupled,
        x_basis,
        _x_transform,
        b_basis,
        _b_label_basis,
        _x_matrices,
        _b_matrices,
    ) = _build_hamiltonian_sequences(
        np.array([0.0]),
        np.array([0.0, 0.0, 1e-3]),
        [0],
        [1],
    )

    serial = _build_e1_operator_matrices(x_basis, b_basis, workers=1)
    parallel = _build_e1_operator_matrices(x_basis, b_basis, workers=2)

    for name in serial:
        assert np.allclose(serial[name], parallel[name])


def test_block_tracked_manifold_matches_serial_tracking():
    matrices = []
    for field in np.linspace(-1.0, 1.0, 9):
        matrices.append(np.array([[field, 0.05], [0.05, -field]], dtype=float))
    matrices_array = np.asarray(matrices, dtype=np.complex128)

    serial_energies, serial_vectors, serial_overlaps = _build_tracked_manifold(
        matrices_array,
        workers=1,
    )
    parallel_energies, parallel_vectors, parallel_overlaps = _build_tracked_manifold(
        matrices_array,
        workers=3,
    )

    assert np.allclose(serial_energies, parallel_energies)
    assert np.allclose(serial_overlaps, parallel_overlaps)
    vector_overlaps = np.abs(np.sum(serial_vectors.conj() * parallel_vectors, axis=1))
    assert np.allclose(vector_overlaps, 1.0)


def test_tracking_boundary_permutation_uses_energy_distance():
    reference_energies = np.array([0.0, 10.0])
    candidate_energies = np.array([9.0, 1.0])
    reference_vectors = np.eye(2, dtype=np.complex128)
    candidate_vectors = np.array(
        [
            [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)],
            [1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0)],
        ],
        dtype=np.complex128,
    )

    permutation = _tracking_boundary_permutation(
        reference_energies,
        reference_vectors,
        candidate_energies,
        candidate_vectors,
    )

    assert np.array_equal(permutation, np.array([1, 0]))


def test_tiny_grid_parallel_tracking_matches_serial_frequencies():
    serial = build_transition_grid(
        ez_values=[0.0, 25.0, 50.0],
        ground_js=[0, 1],
        excited_js=[1],
        progress=False,
        tracking_workers=1,
    )
    parallel = build_transition_grid(
        ez_values=[0.0, 25.0, 50.0],
        ground_js=[0, 1],
        excited_js=[1],
        progress=False,
        tracking_workers=2,
    )

    assert parallel.metadata["tracking_workers"] == 2
    for ez_v_cm in serial.ez_values:
        serial_lines = serial.closest_slice(float(ez_v_cm)).lines
        parallel_lines = parallel.closest_slice(float(ez_v_cm)).lines
        assert len(serial_lines) == len(parallel_lines)
        assert np.allclose(
            serial_lines["frequency_ir_mhz"].to_numpy(),
            parallel_lines["frequency_ir_mhz"].to_numpy(),
        )


def test_eigenshuffle_keeps_branch_identity_through_crossing():
    matrices = []
    for field in np.linspace(-1.0, 1.0, 9):
        matrices.append(np.array([[field, 0.05], [0.05, -field]], dtype=float))
    eigenvalues, eigenvectors = eigenshuffle_eigh(np.asarray(matrices))

    overlaps = np.abs(np.sum(eigenvectors[:-1].conj() * eigenvectors[1:], axis=1))

    assert eigenvalues.shape == (9, 2)
    assert np.all(overlaps > 0.7)


def test_cluster_transition_components_groups_within_transition_name():
    lines = pd.DataFrame(
        [
            {
                "transition_name": "R(0) F1'=1/2 F'=1",
                "branch": "R",
                "J_ground": 0,
                "F1_ground": 0.5,
                "F_ground": 1,
                "frequency_ir_mhz": 100.0,
                "strength": 1.0,
                "nphotons": 10.0,
                "ground_mJ_dominant": 0,
                "mF_ground": 0,
                "mF_excited": 0,
            },
            {
                "transition_name": "R(0) F1'=1/2 F'=1",
                "branch": "R",
                "J_ground": 0,
                "F1_ground": 0.5,
                "F_ground": 1,
                "frequency_ir_mhz": 101.0,
                "strength": 3.0,
                "nphotons": 20.0,
                "ground_mJ_dominant": 1,
                "mF_ground": 0,
                "mF_excited": 1,
            },
            {
                "transition_name": "Q(1) F1'=1/2 F'=1",
                "branch": "Q",
                "J_ground": 1,
                "F1_ground": 1.5,
                "F_ground": 2,
                "frequency_ir_mhz": 101.5,
                "strength": 5.0,
                "nphotons": 30.0,
                "ground_mJ_dominant": 1,
                "mF_ground": -1,
                "mF_excited": -1,
            },
            {
                "transition_name": "R(0) F1'=1/2 F'=1",
                "branch": "R",
                "J_ground": 0,
                "F1_ground": 0.5,
                "F_ground": 1,
                "frequency_ir_mhz": 110.0,
                "strength": 1.0,
                "nphotons": 10.0,
                "ground_mJ_dominant": -1,
                "mF_ground": 0,
                "mF_excited": -1,
            },
        ]
    )

    clusters, cluster_lines = cluster_transition_components(
        lines,
        resolving_frequency_mhz=2.5,
        include_cluster_lines=True,
    )

    assert len(clusters) == 3
    assert len(cluster_lines) == len(lines)
    first = clusters[clusters["transition_name"] == "R(0) F1'=1/2 F'=1"].iloc[0]
    assert first["components"] == 2
    assert first["mf_branches"] == 2
    assert first["delta_mf"] == "0, +1"
    assert first["frequency_ir_mhz"] == 100.75
    assert "mf_dominant" not in clusters.columns
    assert "ground_mJ_dominant" in cluster_lines.columns
    assert "F1_ground" in cluster_lines.columns
    assert "F_ground" in cluster_lines.columns
    assert "nphotons" in cluster_lines.columns


def test_enrich_cluster_display_fields_adds_lazy_mf_summaries():
    lines = pd.DataFrame(
        [
            {
                "transition_name": "R(0) F1'=1/2 F'=1",
                "branch": "R",
                "J_ground": 0,
                "frequency_ir_mhz": 100.0,
                "strength": 1.0,
                "nphotons": 10.0,
                "mF_ground": 0,
                "mF_excited": 0,
            },
            {
                "transition_name": "R(0) F1'=1/2 F'=1",
                "branch": "R",
                "J_ground": 0,
                "frequency_ir_mhz": 101.0,
                "strength": 3.0,
                "nphotons": 20.0,
                "mF_ground": 0,
                "mF_excited": 1,
            },
        ]
    )

    clusters, cluster_lines = cluster_transition_components(
        lines,
        resolving_frequency_mhz=2.5,
        include_cluster_lines=True,
    )
    enriched = enrich_cluster_display_fields(
        clusters,
        cluster_lines,
        include_full_detail=True,
    )

    cluster = enriched.iloc[0]
    assert cluster["mf_dominant"] == "0 -> 1 (75%), 0 -> 0 (25%)"
    assert cluster["mf_detail"] == "0 -> 1 (75%, x1), 0 -> 0 (25%, x1)"


def test_cluster_transition_components_splits_by_excited_parent_parity():
    lines = pd.DataFrame(
        [
            {
                "transition_name": "R(2) F1'=7/2 F'=3",
                "branch": "R",
                "J_ground": 2,
                "frequency_ir_mhz": 100.0,
                "strength": 1.0,
                "nphotons": 10.0,
                "mF_ground": 0,
                "mF_excited": 0,
                "excited_label": "|B, J = 3, F1 = 7/2, F = 3, mF = 0, P = -, Omega = 1>",
            },
            {
                "transition_name": "R(2) F1'=7/2 F'=3",
                "branch": "R",
                "J_ground": 2,
                "frequency_ir_mhz": 100.1,
                "strength": 1.0,
                "nphotons": 10.0,
                "mF_ground": 0,
                "mF_excited": 0,
                "excited_label": "|B, J = 3, F1 = 7/2, F = 3, mF = 0, P = +, Omega = 1>",
            },
        ]
    )

    clusters = cluster_transition_components(lines, resolving_frequency_mhz=2.5)

    assert len(clusters) == 2
    assert set(clusters["excited_parent_parity"]) == {"-", "+"}


def test_cluster_transition_components_merges_same_parity_parent_labels():
    lines = pd.DataFrame(
        [
            {
                "transition_name": "R(2) F1'=5/2 F'=2",
                "branch": "R",
                "J_ground": 2,
                "frequency_ir_mhz": 100.0000,
                "strength": 1.0,
                "nphotons": 10.0,
                "mF_ground": 1,
                "mF_excited": 2,
                "excited_label": "|B, J = 3, F1 = 5/2, F = 2, mF = 2, P = -, Omega = 1>",
            },
            {
                "transition_name": "R(2) F1'=5/2 F'=2",
                "branch": "R",
                "J_ground": 2,
                "frequency_ir_mhz": 100.0005,
                "strength": 1.0,
                "nphotons": 10.0,
                "mF_ground": 0,
                "mF_excited": 1,
                "excited_label": "|B, J = 3, F1 = 5/2, F = 2, mF = 1, P = -, Omega = 1>",
            },
        ]
    )

    clusters, cluster_lines = cluster_transition_components(
        lines,
        resolving_frequency_mhz=2.5,
        include_cluster_lines=True,
    )

    assert len(clusters) == 1
    cluster = clusters.iloc[0]
    assert cluster["excited_parent_parity"] == "-"
    assert cluster["components"] == 2
    assert cluster["mf_branches"] == 2

    enriched = enrich_cluster_display_fields(
        clusters,
        cluster_lines,
        include_full_detail=True,
    )
    assert "mF = 2" in enriched.iloc[0]["excited_parent_label"]
    assert "mF = 1" in enriched.iloc[0]["excited_parent_label"]


def test_generate_cluster_dataframe_includes_parent_and_mf_summary_columns():
    clusters = pd.DataFrame(
        [
            {
                "transition_name": "R(0) F1'=1/2 F'=1",
                "branch": "R",
                "J_ground": 0,
                "excited_parent_parity": "-",
                "excited_parent_label": "|B, J = 1, F1 = 1/2, F = 1, mF = 0, P = -, Omega = 1>",
                "frequency_ir_mhz": 100.0,
                "spread_ir_mhz": 0.0,
                "strength": 1.0,
                "components": 1,
                "mf_branches": 1,
                "delta_mf": "0",
                "mf_dominant": "0 -> 0 (100%)",
                "mf_detail": "0 -> 0 (100%, x1)",
                "mF": "0 -> 0",
                "nphotons": 10.0,
            },
            {
                "transition_name": "R(0) F1'=1/2 F'=1",
                "branch": "R",
                "J_ground": 0,
                "excited_parent_parity": "+",
                "excited_parent_label": "|B, J = 1, F1 = 1/2, F = 1, mF = 0, P = +, Omega = 1>",
                "frequency_ir_mhz": 104.0,
                "spread_ir_mhz": 0.0,
                "strength": 3.0,
                "components": 1,
                "mf_branches": 1,
                "delta_mf": "+1",
                "mf_dominant": "0 -> 1 (100%)",
                "mf_detail": "0 -> 1 (100%, x1)",
                "mF": "0 -> 1",
                "nphotons": 20.0,
            },
        ]
    )

    df = generate_cluster_dataframe(
        clusters,
        "R(0) F1'=1/2 F'=1",
        energy_lim=(-20, 20),
        ir_uv="IR",
        calibration_offset_ir_mhz=1.0,
    )

    assert list(df["P'"]) == ["-", "+"]
    assert list(df["state pairs"]) == [1, 1]
    assert list(df["mF branches"]) == [1, 1]
    assert list(df["dominant mF"]) == ["0 -> 0 (100%)", "0 -> 1 (100%)"]
    assert list(df["excited parent"])[0].endswith("P = -, Omega = 1>")


def test_generate_cluster_dataframe_uses_weighted_reference_centroid():
    clusters = pd.DataFrame(
        [
            {
                "transition_name": "R(0) F1'=1/2 F'=1",
                "branch": "R",
                "J_ground": 0,
                "excited_parent_parity": "-",
                "excited_parent_label": "|B, J = 1, F1 = 1/2, F = 1, mF = 0, P = -, Omega = 1>",
                "frequency_ir_mhz": 100.0,
                "spread_ir_mhz": 0.0,
                "strength": 1.0,
                "components": 1,
                "mf_branches": 1,
                "delta_mf": "0",
                "mf_dominant": "0 -> 0 (100%)",
                "mf_detail": "0 -> 0 (100%, x1)",
                "mF": "0 -> 0",
                "nphotons": 10.0,
            },
            {
                "transition_name": "R(0) F1'=1/2 F'=1",
                "branch": "R",
                "J_ground": 0,
                "excited_parent_parity": "+",
                "excited_parent_label": "|B, J = 1, F1 = 1/2, F = 1, mF = 1, P = +, Omega = 1>",
                "frequency_ir_mhz": 104.0,
                "spread_ir_mhz": 0.0,
                "strength": 3.0,
                "components": 1,
                "mf_branches": 1,
                "delta_mf": "+1",
                "mf_dominant": "0 -> 1 (100%)",
                "mf_detail": "0 -> 1 (100%, x1)",
                "mF": "0 -> 1",
                "nphotons": 20.0,
            },
        ]
    )

    centroid = transition_centroid(clusters, "R(0) F1'=1/2 F'=1")
    df = generate_cluster_dataframe(
        clusters,
        "R(0) F1'=1/2 F'=1",
        energy_lim=(-20, 20),
        ir_uv="IR",
        calibration_offset_ir_mhz=1.0,
        zero_field_clusters=clusters.assign(frequency_ir_mhz=[95.0, 103.0]),
    )

    assert centroid == 103.0
    assert list(df["Δ freq [IR, MHz]"]) == [-3.0, 1.0]
    assert list(df["Δ from 0 V/cm [IR, MHz]"]) == [-1.0, 3.0]
    assert list(df["frequency [IR, GHz]"]) == [0.101, 0.105]


def test_generate_cluster_dataframe_applies_reference_axis_shift():
    clusters = pd.DataFrame(
        [
            {
                "transition_name": "R(0) F1'=1/2 F'=1",
                "branch": "R",
                "J_ground": 0,
                "excited_parent_parity": "-",
                "frequency_ir_mhz": 100.0,
                "spread_ir_mhz": 0.0,
                "strength": 1.0,
                "components": 1,
                "mf_branches": 1,
                "delta_mf": "0",
                "nphotons": 10.0,
            },
            {
                "transition_name": "R(0) F1'=1/2 F'=1",
                "branch": "R",
                "J_ground": 0,
                "excited_parent_parity": "+",
                "frequency_ir_mhz": 104.0,
                "spread_ir_mhz": 0.0,
                "strength": 3.0,
                "components": 1,
                "mf_branches": 1,
                "delta_mf": "+1",
                "nphotons": 20.0,
            },
        ]
    )

    df = generate_cluster_dataframe(
        clusters,
        "R(0) F1'=1/2 F'=1",
        energy_lim=(-20, 100),
        ir_uv="UV",
        calibration_offset_ir_mhz=1.0,
        reference_axis_shift_ir_mhz=20.0,
        zero_field_clusters=clusters.assign(frequency_ir_mhz=[95.0, 103.0]),
    )

    delta = chr(916)
    assert list(df[f"{delta} freq [UV, MHz]"]) == [-12.0, 4.0]
    assert list(df[f"{delta} from 0 V/cm [UV, MHz]"]) == [-4.0, 12.0]
    assert list(df["frequency [UV, GHz]"]) == [0.484, 0.5]


def test_generate_cluster_dataframe_omits_zero_field_shift_without_reference():
    clusters = pd.DataFrame(
        [
            {
                "transition_name": "R(0) F1'=1/2 F'=1",
                "branch": "R",
                "J_ground": 0,
                "excited_parent_parity": "+",
                "frequency_ir_mhz": 100.0,
                "spread_ir_mhz": 0.0,
                "strength": 1.0,
                "components": 1,
                "mf_branches": 1,
                "delta_mf": "0",
                "nphotons": 10.0,
            },
        ]
    )

    df = generate_cluster_dataframe(
        clusters,
        "R(0) F1'=1/2 F'=1",
        energy_lim=(-20, 20),
        ir_uv="IR",
    )

    assert "Δ freq [IR, MHz]" in df.columns
    assert "Δ from 0 V/cm [IR, MHz]" not in df.columns


def test_line_records_convert_uv_hamiltonian_frequency_to_ir():
    component_strengths = {
        "all": np.array([[1.0]]),
        "x": np.array([[0.25]]),
        "y": np.array([[0.25]]),
        "z": np.array([[0.50]]),
        "sigma_plus": np.array([[0.10]]),
        "sigma_minus": np.array([[0.40]]),
    }
    records = _line_records_for_slice(
        ez_v_cm=0.0,
        x_energies_mhz=np.array([0.0]),
        b_energies_mhz=np.array([400.0]),
        strength_matrix=np.array([[1.0]]),
        component_strengths=component_strengths,
        candidates=[
            {
                "transition_name": "R(0) F1'=1/2 F'=1",
                "branch": "R",
                "J_ground": 0,
                "ground_state_id": 0,
                "excited_state_id": 0,
            }
        ],
        ground_js=np.array([0]),
        x_largest_current=["X"],
        x_mj_dominant=np.array([0]),
        b_largest_current=["B"],
        coupling_cutoff=0.0,
    )

    assert records[0]["frequency_ir_mhz"] == 100.0
    assert records[0]["ground_mJ_dominant"] == 0


def test_polarization_strength_matrices_include_coherent_sigma_components():
    x = np.array([[1.0 + 0.0j, 1.0 + 0.0j]])
    y = np.array([[0.0 + 1.0j, 0.0 - 1.0j]])
    z = np.array([[2.0 + 0.0j, 0.0 + 0.0j]])
    strengths = _polarization_strength_matrices({"x": x, "y": y, "z": z})

    assert np.allclose(strengths["all"], [[6.0, 2.0]])
    assert np.allclose(strengths["sigma_plus"], [[2.0, 0.0]])
    assert np.allclose(strengths["sigma_minus"], [[0.0, 2.0]])


def test_apply_polarization_selection_reweights_and_filters_lines():
    lines = pd.DataFrame(
        [
            {"strength": 3.0, "strength_all": 3.0, "strength_z": 0.0},
            {"strength": 1.0, "strength_all": 1.0, "strength_z": 0.5},
        ]
    )

    selected = apply_polarization_selection(lines, "Z", coupling_cutoff=1e-14)

    assert list(POLARIZATION_STRENGTH_COLUMNS) == [
        "All",
        "X",
        "Y",
        "Z",
        "sigma+",
        "sigma-",
    ]
    assert len(selected) == 1
    assert selected.iloc[0]["strength"] == 0.5
    assert selected.iloc[0]["polarization"] == "Z"


def test_dominant_uncoupled_mj_values_uses_largest_uncoupled_component():
    class Uncoupled:
        def __init__(self, mJ: int) -> None:
            self.mJ = mJ

    eigenvectors = np.array(
        [
            [0.1, 0.2],
            [0.9, 0.1],
            [0.2, 0.8],
        ],
        dtype=np.complex128,
    )
    transform = np.eye(3, dtype=np.complex128)
    uncoupled_basis = [Uncoupled(-1), Uncoupled(0), Uncoupled(1)]

    mjs = _dominant_uncoupled_mj_values(eigenvectors, transform, uncoupled_basis)

    assert list(mjs) == [0, 1]


def test_operator_matrix_couplings_match_direct_mixed_state_calls():
    (
        _x_uncoupled,
        x_basis,
        _x_transform,
        b_basis,
        _b_label_basis,
        x_matrices,
        b_matrices,
    ) = (
        _build_hamiltonian_sequences(
            np.array([25.0]),
            np.array([0.0, 0.0, 1e-3]),
            [0, 1],
            [1],
        )
    )
    _, x_eigenvectors, _ = _build_tracked_manifold(x_matrices)
    _, b_eigenvectors, _ = _build_tracked_manifold(b_matrices)
    e1_operators = _build_e1_operator_matrices(x_basis, b_basis)
    coupling_matrices = _coupling_matrices_for_field(
        e1_operators,
        x_eigenvectors[0],
        b_eigenvectors[0],
    )
    x_states = hamiltonian.matrix_to_states(x_eigenvectors[0], x_basis)
    b_states = hamiltonian.matrix_to_states(b_eigenvectors[0], b_basis)

    for pol_name, pol_vec in E1_POLARIZATION_VECTORS.items():
        direct = generate_ED_ME_mixed_state(
            b_states[-1],
            x_states[-1],
            pol_vec=pol_vec,
            reduced=False,
        )
        assert np.isclose(direct, coupling_matrices[pol_name][-1, -1])


def test_candidate_generation_keeps_same_parity_mixed_field_candidates():
    (
        _x_uncoupled,
        x_basis,
        _x_transform,
        b_basis,
        b_label_basis,
        x_matrices,
        b_matrices,
    ) = (
        _build_hamiltonian_sequences(
            np.array([0.0]),
            np.array([0.0, 0.0, 1e-3]),
            [0, 1],
            [1],
        )
    )
    _, x_eigenvectors, _ = _build_tracked_manifold(x_matrices)
    _, b_eigenvectors, _ = _build_tracked_manifold(b_matrices)
    x_labels, _, _ = _assign_labels_by_overlap(x_eigenvectors[0], x_basis, x_basis)
    b_labels, _, _ = _assign_labels_by_overlap(
        b_eigenvectors[0], b_label_basis, b_basis
    )
    candidates = _build_transition_candidates(x_labels, b_labels)

    assert all("F1_ground" in candidate for candidate in candidates)
    assert all("F_ground" in candidate for candidate in candidates)
    first_candidate = candidates[0]
    first_ground_label = x_labels[int(first_candidate["ground_state_id"])]
    assert first_candidate["F1_ground"] == float(first_ground_label.F1)
    assert first_candidate["F_ground"] == int(first_ground_label.F)
    assert any(
        x_labels[candidate["ground_state_id"]].P
        == b_labels[candidate["excited_state_id"]].P
        for candidate in candidates
    )


def test_tiny_grid_precompute_filters_forbidden_mf_and_e2():
    grid = build_transition_grid(
        ez_values=[0.0],
        ground_js=[0, 1],
        excited_js=[1],
        progress=False,
    )

    grid_slice = grid.closest_slice(0.0)

    assert grid.metadata["schema_version"] == 6
    assert len(grid_slice.lines) > 0
    assert set(grid_slice.lines["branch"]).issubset({"P", "Q", "R"})
    assert np.all(np.abs(grid_slice.lines["delta_mF"].to_numpy()) <= 1)
    assert np.allclose(
        grid_slice.lines[["strength_x", "strength_y", "strength_z"]].sum(axis=1),
        grid_slice.lines["strength"],
    )
    assert np.allclose(grid_slice.lines["strength_all"], grid_slice.lines["strength"])
    assert "strength_sigma_plus" in grid_slice.lines.columns
    assert "strength_sigma_minus" in grid_slice.lines.columns
    assert "ground_mJ_dominant" in grid_slice.lines.columns
    assert "F1_ground" in grid_slice.lines.columns
    assert "F_ground" in grid_slice.lines.columns
    assert "mF_ground" in grid_slice.lines.columns
    assert "mF_excited" in grid_slice.lines.columns
    assert "excited_parent_parity" in grid_slice.lines.columns
