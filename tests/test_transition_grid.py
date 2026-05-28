import sys
from pathlib import Path

import numpy as np
import pandas as pd
from centrex_tlf import hamiltonian
from eigenshuffle import eigenshuffle_eigh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transition_grid import (
    E1_POLARIZATION_VECTORS,
    _assign_labels_by_overlap,
    _build_e1_operator_matrices,
    _build_hamiltonian_sequences,
    _build_tracked_manifold,
    _build_transition_candidates,
    _coupling_matrices_for_field,
    _line_records_for_slice,
    build_transition_grid,
    cluster_transition_components,
    enrich_cluster_display_fields,
    generate_cluster_dataframe,
    generate_ED_ME_mixed_state,
    transition_centroid,
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
            {
                "transition_name": "Q(1) F1'=1/2 F'=1",
                "branch": "Q",
                "J_ground": 1,
                "frequency_ir_mhz": 101.5,
                "strength": 5.0,
                "nphotons": 30.0,
                "mF_ground": -1,
                "mF_excited": -1,
            },
            {
                "transition_name": "R(0) F1'=1/2 F'=1",
                "branch": "R",
                "J_ground": 0,
                "frequency_ir_mhz": 110.0,
                "strength": 1.0,
                "nphotons": 10.0,
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
    )

    assert centroid == 103.0
    assert list(df["delta frequency [IR, MHz]"]) == [-3.0, 1.0]
    assert list(df["frequency [IR, GHz]"]) == [0.101, 0.105]


def test_line_records_convert_uv_hamiltonian_frequency_to_ir():
    component_strengths = {
        "x": np.array([[0.25]]),
        "y": np.array([[0.25]]),
        "z": np.array([[0.50]]),
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
        b_largest_current=["B"],
        coupling_cutoff=0.0,
    )

    assert records[0]["frequency_ir_mhz"] == 100.0


def test_operator_matrix_couplings_match_direct_mixed_state_calls():
    x_basis, b_basis, _b_label_basis, x_matrices, b_matrices = (
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
    x_basis, b_basis, b_label_basis, x_matrices, b_matrices = (
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

    assert len(grid_slice.lines) > 0
    assert set(grid_slice.lines["branch"]).issubset({"P", "Q", "R"})
    assert np.all(np.abs(grid_slice.lines["delta_mF"].to_numpy()) <= 1)
    assert np.allclose(
        grid_slice.lines[["strength_x", "strength_y", "strength_z"]].sum(axis=1),
        grid_slice.lines["strength"],
    )
    assert "mF_ground" in grid_slice.lines.columns
    assert "mF_excited" in grid_slice.lines.columns
    assert "excited_parent_parity" in grid_slice.lines.columns
