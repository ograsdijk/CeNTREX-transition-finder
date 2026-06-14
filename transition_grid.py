import argparse
import pickle
import re
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import shared_memory
from pathlib import Path
from typing import Callable, Literal, Sequence, TextIO, overload

import numpy as np
import numpy.typing as npt
import pandas as pd
from centrex_tlf import hamiltonian, states, transitions
from centrex_tlf.hamiltonian.basis_transformations import generate_transform_matrix
from centrex_tlf.states.generate_states import generate_coupled_states_ground
from centrex_tlf.states.states import CoupledBasisState
from eigenshuffle import eigenshuffle_eigh
from scipy.optimize import linear_sum_assignment

try:
    from centrex_tlf.hamiltonian.matrix_elements_electric_dipole import (
        generate_ED_ME_mixed_state,
    )
except ModuleNotFoundError:
    from centrex_tlf.hamiltonian.matrix_elements import generate_ED_ME_mixed_state

GRID_ARTIFACT = "transition_grid.pkl"
DEFAULT_EZ_VALUES = np.arange(0.0, 500.0 + 25.0, 25.0)
DEFAULT_B_FIELD = np.array([0.0, 0.0, 1e-3], dtype=float)
UV_TO_IR_FREQUENCY = 0.25
DEFAULT_COUPLING_CUTOFF = 1e-14
POLARIZATION_STRENGTH_COLUMNS = {
    "All": "strength_all",
    "X": "strength_x",
    "Y": "strength_y",
    "Z": "strength_z",
    "sigma+": "strength_sigma_plus",
    "sigma-": "strength_sigma_minus",
}
E1_POLARIZATION_VECTORS = {
    "x": np.array([1.0, 0.0, 0.0], dtype=complex),
    "y": np.array([0.0, 1.0, 0.0], dtype=complex),
    "z": np.array([0.0, 0.0, 1.0], dtype=complex),
}
PARENT_PARITY_PATTERN = re.compile(r"P\s*=\s*([+-])")


def format_elapsed(seconds: float) -> str:
    seconds_int = max(0, int(seconds))
    hours, remainder = divmod(seconds_int, 3600)
    minutes, seconds_int = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_int:02d}"


class ElapsedSpinner:
    def __init__(
        self,
        message: str,
        *,
        enabled: bool = True,
        stream: TextIO = sys.stdout,
        interval_s: float = 0.1,
    ) -> None:
        self.message = message
        self.enabled = bool(enabled and stream.isatty())
        self.stream = stream
        self.interval_s = interval_s
        self._frames = ("|", "/", "-", "\\")
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._start = 0.0
        self._last_len = 0

    def __enter__(self) -> "ElapsedSpinner":
        if not self.enabled:
            return self
        self._start = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        if not self.enabled:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        with self._lock:
            self._clear_line()
            self.stream.flush()

    def print_line(self, line: str) -> None:
        if not self.enabled:
            print(line)
            return
        with self._lock:
            self._clear_line()
            print(line, file=self.stream)
            self._write_frame(self._frames[0])
            self.stream.flush()

    def _run(self) -> None:
        frame_idx = 0
        while not self._stop.is_set():
            with self._lock:
                self._write_frame(self._frames[frame_idx % len(self._frames)])
                self.stream.flush()
            frame_idx += 1
            self._stop.wait(self.interval_s)

    def _write_frame(self, frame: str) -> None:
        line = (
            f"{self.message} {frame} elapsed "
            f"{format_elapsed(time.perf_counter() - self._start)}"
        )
        self.stream.write("\r" + line)
        if self._last_len > len(line):
            self.stream.write(" " * (self._last_len - len(line)))
        self._last_len = len(line)

    def _clear_line(self) -> None:
        if self._last_len:
            self.stream.write("\r" + " " * self._last_len + "\r")
            self._last_len = 0


@dataclass
class TransitionGridSlice:
    ez_v_cm: float
    lines: pd.DataFrame
    timings: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, float | int] = field(default_factory=dict)


@dataclass
class TransitionGrid:
    ez_values: npt.NDArray[np.float64]
    slices: dict[float, TransitionGridSlice]
    metadata: dict[str, object] = field(default_factory=dict)

    def closest_slice(self, ez_v_cm: float) -> TransitionGridSlice:
        idx = int(np.argmin(np.abs(self.ez_values - ez_v_cm)))
        return self.slices[float(self.ez_values[idx])]

    @property
    def transition_names(self) -> list[str]:
        names: set[str] = set()
        for grid_slice in self.slices.values():
            names.update(grid_slice.lines["transition_name"].unique())
        return sorted(names)


def load_transition_grid(path: str | Path = GRID_ARTIFACT) -> TransitionGrid:
    with open(path, "rb") as f:
        grid: TransitionGrid = pickle.load(f)

    for grid_slice in grid.slices.values():
        lines = grid_slice.lines
        if lines.empty:
            continue
        if (
            "excited_parent_parity" not in lines.columns
            and "excited_label" in lines.columns
        ):
            grid_slice.lines = lines.assign(
                excited_parent_parity=lines["excited_label"].map(
                    _parent_parity_from_label
                )
            )
    return grid


def save_transition_grid(
    grid: TransitionGrid, path: str | Path = GRID_ARTIFACT
) -> None:
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(grid, f)
    tmp_path.replace(path)


def apply_polarization_selection(
    lines: pd.DataFrame,
    polarization: str,
    coupling_cutoff: float = DEFAULT_COUPLING_CUTOFF,
) -> pd.DataFrame:
    if polarization not in POLARIZATION_STRENGTH_COLUMNS:
        options = ", ".join(POLARIZATION_STRENGTH_COLUMNS)
        raise ValueError(f"Unknown polarization {polarization!r}; expected one of {options}.")
    if lines.empty:
        return lines.copy()

    strength_column = POLARIZATION_STRENGTH_COLUMNS[polarization]
    if strength_column not in lines.columns:
        raise ValueError(
            f"Grid does not contain {strength_column!r}. Rebuild {GRID_ARTIFACT}."
        )

    selected = lines.copy()
    selected["strength"] = selected[strength_column].to_numpy(dtype=float)
    selected = selected[selected["strength"] > coupling_cutoff].copy()
    selected["polarization"] = polarization
    return selected.reset_index(drop=True)


def _field_vector(ez_v_cm: float) -> npt.NDArray[np.float64]:
    return np.array([0.0, 0.0, float(ez_v_cm)], dtype=float)


def _state_label(state: CoupledBasisState) -> dict[str, object]:
    return {
        "label": state.state_string(),
        "J": int(state.J),
        "F1": float(state.F1),
        "F": int(state.F),
        "mF": int(state.mF),
        "P": int(state.P) if state.P is not None else None,
    }


def _parity_symbol(value: object) -> str:
    if value in {1, "+"}:
        return "+"
    if value in {-1, "-"}:
        return "-"
    return "?"


def _largest_basis_labels(
    eigenvectors: npt.NDArray[np.complex128],
    basis: Sequence[CoupledBasisState],
) -> list[CoupledBasisState]:
    return [
        basis[int(np.argmax(np.abs(eigenvectors[:, idx])))]
        for idx in range(eigenvectors.shape[1])
    ]


def _state_strings(labels: Sequence[CoupledBasisState]) -> list[str]:
    return [label.state_string() for label in labels]


def _tracking_overlap(
    previous: npt.NDArray[np.complex128],
    current: npt.NDArray[np.complex128],
) -> npt.NDArray[np.float64]:
    overlaps = np.sum(previous.conj() * current, axis=0)
    return np.abs(overlaps).astype(np.float64)


def _column_permutation(
    reference: npt.NDArray[np.complex128],
    candidate: npt.NDArray[np.complex128],
) -> npt.NDArray[np.int_]:
    overlap = np.abs(reference.conj().T @ candidate).astype(np.float64)
    row_ids, column_ids = linear_sum_assignment(-overlap)
    return column_ids[np.argsort(row_ids)].astype(int)


def _tracking_boundary_permutation(
    reference_energies: npt.NDArray[np.float64],
    reference_vectors: npt.NDArray[np.complex128],
    candidate_energies: npt.NDArray[np.float64],
    candidate_vectors: npt.NDArray[np.complex128],
) -> npt.NDArray[np.int_]:
    overlap_distance = 1.0 - np.abs(reference_vectors.conj().T @ candidate_vectors)
    energy_distance = np.abs(
        reference_energies[:, np.newaxis] - candidate_energies[np.newaxis, :]
    )
    row_ids, column_ids = linear_sum_assignment(overlap_distance * energy_distance)
    return column_ids[np.argsort(row_ids)].astype(int)


TrackingBlockTask = tuple[int, int, int, str, tuple[int, ...], str]


def _tracked_block_worker(
    task: TrackingBlockTask,
) -> tuple[int, int, int, npt.NDArray[np.float64], npt.NDArray[np.complex128]]:
    block_id, start, stop, shared_name, shape, dtype_str = task
    shared = shared_memory.SharedMemory(name=shared_name)
    matrices: npt.NDArray[np.complex128] | None = None
    try:
        matrices = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shared.buf)
        energies, eigenvectors = eigenshuffle_eigh(
            matrices[start:stop],
            use_eigenvalues=True,
        )
        return block_id, start, stop, energies, eigenvectors
    finally:
        del matrices
        shared.close()


def _make_shared_array(
    values: npt.NDArray[np.complex128],
) -> tuple[shared_memory.SharedMemory, npt.NDArray[np.complex128]]:
    contiguous = np.ascontiguousarray(values)
    shared = shared_memory.SharedMemory(create=True, size=contiguous.nbytes)
    shared_values = np.ndarray(
        contiguous.shape,
        dtype=contiguous.dtype,
        buffer=shared.buf,
    )
    shared_values[:] = contiguous
    return shared, shared_values


def _tracking_blocks(n_matrices: int, workers: int) -> list[tuple[int, int, int]]:
    n_blocks = min(max(1, int(workers)), n_matrices)
    index_blocks = np.array_split(np.arange(n_matrices), n_blocks)
    blocks: list[tuple[int, int, int]] = []
    for block_id, indices in enumerate(index_blocks):
        if len(indices) == 0:
            continue
        blocks.append((block_id, int(indices[0]), int(indices[-1]) + 1))
    return blocks


def _build_tracked_manifold_serial(
    matrices: npt.NDArray[np.complex128],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.complex128]]:
    return eigenshuffle_eigh(matrices, use_eigenvalues=True)


def _build_tracked_manifold(
    matrices: npt.NDArray[np.complex128],
    workers: int = 1,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.complex128],
    npt.NDArray[np.float64],
]:
    workers = max(1, int(workers))
    if workers == 1 or len(matrices) < 2:
        energies, eigenvectors = _build_tracked_manifold_serial(matrices)
    else:
        blocks = _tracking_blocks(len(matrices), workers)
        shared, shared_matrices = _make_shared_array(matrices)
        try:
            tasks = [
                (
                    block_id,
                    start,
                    stop,
                    shared.name,
                    shared_matrices.shape,
                    shared_matrices.dtype.str,
                )
                for block_id, start, stop in blocks
            ]
            with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
                results = list(pool.map(_tracked_block_worker, tasks))
        finally:
            del shared_matrices
            shared.close()
            shared.unlink()
        results = sorted(results, key=lambda result: result[0])
        energy_blocks = [result[3] for result in results]
        eigenvector_blocks = [result[4] for result in results]

        for block_idx in range(1, len(results)):
            current_perm = _tracking_boundary_permutation(
                energy_blocks[block_idx - 1][-1],
                eigenvector_blocks[block_idx - 1][-1],
                energy_blocks[block_idx][0],
                eigenvector_blocks[block_idx][0],
            )
            energy_blocks[block_idx] = energy_blocks[block_idx][:, current_perm]
            eigenvector_blocks[block_idx] = eigenvector_blocks[block_idx][
                :, :, current_perm
            ]

        energies = np.concatenate(energy_blocks, axis=0)
        eigenvectors = np.concatenate(eigenvector_blocks, axis=0)

    if len(energies) < 2:
        overlaps = np.ones((0, energies.shape[1]), dtype=np.float64)
    else:
        overlaps = np.array(
            [
                _tracking_overlap(eigenvectors[idx - 1], eigenvectors[idx])
                for idx in range(1, len(energies))
            ],
            dtype=np.float64,
        )
    return energies / (2 * np.pi * 1e6), eigenvectors, overlaps


def _state_vector_for_label(
    label: CoupledBasisState,
    basis: Sequence[CoupledBasisState],
) -> npt.NDArray[np.complex128]:
    vector = (1 * label).state_vector(basis)
    if np.linalg.norm(vector) == 0 and label.basis == states.Basis.CoupledP:
        vector = label.transform_to_omega_basis().state_vector(basis)
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError(
            f"Could not express label {label.state_string()} in construction basis."
        )
    return vector / norm


def _assign_labels_by_overlap(
    eigenvectors: npt.NDArray[np.complex128],
    label_basis: Sequence[CoupledBasisState],
    construction_basis: Sequence[CoupledBasisState],
) -> tuple[
    list[CoupledBasisState],
    npt.NDArray[np.int_],
    npt.NDArray[np.float64],
]:
    label_vectors = np.asarray(
        [_state_vector_for_label(label, construction_basis) for label in label_basis],
        dtype=np.complex128,
    )
    overlap = np.abs(label_vectors.conj() @ eigenvectors).astype(np.float64)
    label_ids, state_ids = linear_sum_assignment(-overlap)
    state_ids_by_label = np.full(len(label_basis), -1, dtype=int)
    overlaps_by_label = np.zeros(len(label_basis), dtype=np.float64)
    for label_id, state_id in zip(label_ids, state_ids):
        state_ids_by_label[int(label_id)] = int(state_id)
        overlaps_by_label[int(label_id)] = overlap[int(label_id), int(state_id)]
    if np.any(state_ids_by_label < 0):
        raise ValueError("Could not assign all eigenvectors to zero-field labels.")
    return list(label_basis), state_ids_by_label, overlaps_by_label


def _state_strings_by_overlap(
    eigenvectors: npt.NDArray[np.complex128],
    label_basis: Sequence[CoupledBasisState],
    construction_basis: Sequence[CoupledBasisState],
    label_strings: Sequence[str] | None = None,
) -> list[str]:
    label_vectors = np.asarray(
        [_state_vector_for_label(label, construction_basis) for label in label_basis],
        dtype=np.complex128,
    )
    overlap = np.abs(label_vectors.conj() @ eigenvectors).astype(np.float64)
    label_ids = np.argmax(overlap, axis=0)
    if label_strings is None:
        label_strings = _state_strings(label_basis)
    return [label_strings[int(label_id)] for label_id in label_ids]


def _dominant_uncoupled_mj_values(
    eigenvectors: npt.NDArray[np.complex128],
    coupled_to_uncoupled_transform: npt.NDArray[np.complex128],
    uncoupled_basis: Sequence[object],
) -> npt.NDArray[np.int_]:
    uncoupled_vectors = coupled_to_uncoupled_transform @ eigenvectors
    component_ids = np.argmax(np.abs(uncoupled_vectors), axis=0)
    return np.asarray(
        [int(getattr(uncoupled_basis[int(component_id)], "mJ")) for component_id in component_ids],
        dtype=int,
    )


def _omega_basis_from_parity_labels(
    parity_labels: Sequence[CoupledBasisState],
) -> list[CoupledBasisState]:
    return list(
        states.get_unique_basisstates_from_states(
            [label.transform_to_omega_basis() for label in parity_labels]
        )
    )


def _build_hamiltonian_sequences(
    ez_values: npt.NDArray[np.float64],
    b_field: npt.NDArray[np.float64],
    ground_js: Sequence[int],
    excited_js: Sequence[int],
    excited_j_padding: int = 2,
) -> tuple[
    list[object],
    list[CoupledBasisState],
    npt.NDArray[np.complex128],
    list[CoupledBasisState],
    list[CoupledBasisState],
    npt.NDArray[np.complex128],
    npt.NDArray[np.complex128],
]:
    x_uncoupled = states.generate_uncoupled_states_ground(
        Js=np.asarray(ground_js, dtype=int)
    )
    x_basis = generate_coupled_states_ground(Js=np.asarray(ground_js, dtype=int))
    x_transform = generate_transform_matrix(x_uncoupled, x_basis)
    x_terms = hamiltonian.generate_uncoupled_hamiltonian_X(x_uncoupled)
    x_func = hamiltonian.generate_uncoupled_hamiltonian_X_function(x_terms)

    b_label_basis = states.generate_coupled_states_B(
        states.QuantumSelector(J=list(excited_js), P=[-1, 1]),
        basis=states.Basis.CoupledP,
    )
    b_construct_js = range(1, max(excited_js) + excited_j_padding + 1)
    b_construct_labels = states.generate_coupled_states_B(
        states.QuantumSelector(J=list(b_construct_js), P=[-1, 1]),
        basis=states.Basis.CoupledP,
    )
    b_basis = _omega_basis_from_parity_labels(b_construct_labels)
    b_terms = hamiltonian.generate_coupled_hamiltonian_B(b_basis)
    b_func = hamiltonian.generate_coupled_hamiltonian_B_function(b_terms)

    x_matrices = []
    b_matrices = []
    for ez_v_cm in ez_values:
        e_field = _field_vector(float(ez_v_cm))
        x_matrices.append(x_transform.conj().T @ x_func(e_field, b_field) @ x_transform)
        b_matrices.append(b_func(e_field, b_field))

    return (
        list(x_uncoupled),
        list(x_basis),
        x_transform,
        list(b_basis),
        list(b_label_basis),
        np.asarray(x_matrices),
        np.asarray(b_matrices),
    )


E1OperatorBlockTask = tuple[
    str,
    npt.NDArray[np.complex128],
    int,
    int,
    Sequence[CoupledBasisState],
    Sequence[CoupledBasisState],
]


def _e1_operator_block(
    task: E1OperatorBlockTask,
) -> tuple[str, int, int, npt.NDArray[np.complex128]]:
    name, pol_vec, row_start, row_stop, x_basis, b_basis = task
    block = np.zeros((row_stop - row_start, len(x_basis)), dtype=np.complex128)
    for local_row, excited_id in enumerate(range(row_start, row_stop)):
        excited_state = 1 * b_basis[excited_id]
        for ground_id, ground in enumerate(x_basis):
            block[local_row, ground_id] = generate_ED_ME_mixed_state(
                excited_state,
                1 * ground,
                pol_vec=pol_vec,
                reduced=False,
            )
    return name, row_start, row_stop, block


def _build_e1_operator_matrices_serial(
    x_basis: Sequence[CoupledBasisState],
    b_basis: Sequence[CoupledBasisState],
) -> dict[str, npt.NDArray[np.complex128]]:
    operators: dict[str, npt.NDArray[np.complex128]] = {}
    for name, pol_vec in E1_POLARIZATION_VECTORS.items():
        matrix = np.zeros((len(b_basis), len(x_basis)), dtype=np.complex128)
        for excited_id, excited in enumerate(b_basis):
            excited_state = 1 * excited
            for ground_id, ground in enumerate(x_basis):
                matrix[excited_id, ground_id] = generate_ED_ME_mixed_state(
                    excited_state,
                    1 * ground,
                    pol_vec=pol_vec,
                    reduced=False,
                )
        operators[name] = matrix
    return operators


def _build_e1_operator_matrices(
    x_basis: Sequence[CoupledBasisState],
    b_basis: Sequence[CoupledBasisState],
    workers: int = 1,
) -> dict[str, npt.NDArray[np.complex128]]:
    workers = max(1, int(workers))
    if workers == 1:
        return _build_e1_operator_matrices_serial(x_basis, b_basis)

    operators = {
        name: np.zeros((len(b_basis), len(x_basis)), dtype=np.complex128)
        for name in E1_POLARIZATION_VECTORS
    }
    chunk_count = max(1, workers * 2)
    chunk_size = max(1, (len(b_basis) + chunk_count - 1) // chunk_count)
    tasks: list[E1OperatorBlockTask] = []
    for name, pol_vec in E1_POLARIZATION_VECTORS.items():
        for row_start in range(0, len(b_basis), chunk_size):
            row_stop = min(len(b_basis), row_start + chunk_size)
            tasks.append((name, pol_vec, row_start, row_stop, x_basis, b_basis))

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for name, row_start, row_stop, block in pool.map(_e1_operator_block, tasks):
            operators[name][row_start:row_stop, :] = block
    return operators


def _coupling_matrices_for_field(
    e1_operators: dict[str, npt.NDArray[np.complex128]],
    x_eigenvectors: npt.NDArray[np.complex128],
    b_eigenvectors: npt.NDArray[np.complex128],
) -> dict[str, npt.NDArray[np.complex128]]:
    return {
        name: b_eigenvectors.conj().T @ operator @ x_eigenvectors
        for name, operator in e1_operators.items()
    }


def _incoherent_strength_matrix(
    coupling_matrices: dict[str, npt.NDArray[np.complex128]],
) -> npt.NDArray[np.float64]:
    strength = np.zeros_like(next(iter(coupling_matrices.values())), dtype=np.float64)
    for coupling_matrix in coupling_matrices.values():
        strength += np.abs(coupling_matrix) ** 2
    return strength


def _polarization_strength_matrices(
    coupling_matrices: dict[str, npt.NDArray[np.complex128]],
) -> dict[str, npt.NDArray[np.float64]]:
    x_coupling = coupling_matrices["x"]
    y_coupling = coupling_matrices["y"]
    z_coupling = coupling_matrices["z"]
    sigma_plus_coupling = (-x_coupling + 1j * y_coupling) / np.sqrt(2)
    sigma_minus_coupling = (x_coupling + 1j * y_coupling) / np.sqrt(2)
    return {
        "all": np.abs(x_coupling) ** 2
        + np.abs(y_coupling) ** 2
        + np.abs(z_coupling) ** 2,
        "x": np.abs(x_coupling) ** 2,
        "y": np.abs(y_coupling) ** 2,
        "z": np.abs(z_coupling) ** 2,
        "sigma_plus": np.abs(sigma_plus_coupling) ** 2,
        "sigma_minus": np.abs(sigma_minus_coupling) ** 2,
    }


def _transition_from_zero_labels(
    ground_label: CoupledBasisState,
    excited_label: CoupledBasisState,
) -> transitions.OpticalTransition | None:
    delta_j = int(excited_label.J - ground_label.J)
    if delta_j not in {-1, 0, 1}:
        return None
    try:
        transition_type = transitions.OpticalTransitionType(delta_j)
        return transitions.OpticalTransition(
            transition_type,
            int(ground_label.J),
            float(excited_label.F1),
            int(excited_label.F),
        )
    except ValueError:
        return None


def _build_transition_candidates(
    x_zero_labels: Sequence[CoupledBasisState],
    b_zero_labels: Sequence[CoupledBasisState],
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    ground_metas = [_state_label(label) for label in x_zero_labels]
    excited_metas = [_state_label(label) for label in b_zero_labels]
    for ground_id, ground_label in enumerate(x_zero_labels):
        ground_meta = ground_metas[ground_id]
        for excited_id, excited_label in enumerate(b_zero_labels):
            transition = _transition_from_zero_labels(ground_label, excited_label)
            if transition is None:
                continue
            excited_meta = excited_metas[excited_id]
            delta_mf = int(excited_meta["mF"] - ground_meta["mF"])
            if abs(delta_mf) > 1:
                continue
            candidates.append(
                {
                    "transition_name": transition.name,
                    "branch": transition.name[0],
                    "J_ground": int(transition.J_ground),
                    "F1_ground": float(ground_meta["F1"]),
                    "F_ground": int(ground_meta["F"]),
                    "ground_state_id": int(ground_id),
                    "excited_state_id": int(excited_id),
                    "ground_label": ground_meta["label"],
                    "excited_label": excited_meta["label"],
                    "excited_parent_parity": _parity_symbol(excited_meta["P"]),
                    "mF_ground": ground_meta["mF"],
                    "mF_excited": excited_meta["mF"],
                    "delta_mF": int(delta_mf),
                }
            )
    return candidates


def _branching_by_excited_state(
    decay_matrix: npt.NDArray[np.float64],
    ground_js: npt.NDArray[np.int_],
) -> list[dict[int, float]]:
    unique_ground_js = np.unique(ground_js)
    all_branching: list[dict[int, float]] = []
    for excited_id in range(decay_matrix.shape[0]):
        decay_row = decay_matrix[excited_id]
        total_decay = float(np.sum(decay_row))
        branching: dict[int, float] = {}
        if total_decay > 0:
            for ground_j in unique_ground_js:
                mask = ground_js == ground_j
                branching[int(ground_j)] = float(np.sum(decay_row[mask]) / total_decay)
        all_branching.append(branching)
    return all_branching


def _line_records_for_slice(
    ez_v_cm: float,
    x_energies_mhz: npt.NDArray[np.float64],
    b_energies_mhz: npt.NDArray[np.float64],
    strength_matrix: npt.NDArray[np.float64],
    component_strengths: dict[str, npt.NDArray[np.float64]],
    candidates: Sequence[dict[str, object]],
    ground_js: npt.NDArray[np.int_],
    x_largest_current: Sequence[str],
    x_mj_dominant: npt.NDArray[np.int_],
    b_largest_current: Sequence[str],
    coupling_cutoff: float,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    decay_matrix = strength_matrix
    branching_by_excited = _branching_by_excited_state(decay_matrix, ground_js)

    for candidate in candidates:
        ground_id = int(candidate["ground_state_id"])
        excited_id = int(candidate["excited_state_id"])
        strength = float(strength_matrix[excited_id, ground_id])
        if strength <= coupling_cutoff:
            continue

        branching = branching_by_excited[excited_id]
        target_branching = branching.get(int(candidate["J_ground"]), 0.0)
        nphotons = float(1.0 / (1.0 - 0.99 * target_branching))
        records.append(
            {
                **candidate,
                "ez_v_cm": float(ez_v_cm),
                "frequency_ir_mhz": float(
                    (b_energies_mhz[excited_id] - x_energies_mhz[ground_id])
                    * UV_TO_IR_FREQUENCY
                ),
                "strength": strength,
                "strength_all": float(component_strengths["all"][excited_id, ground_id]),
                "strength_x": float(component_strengths["x"][excited_id, ground_id]),
                "strength_y": float(component_strengths["y"][excited_id, ground_id]),
                "strength_z": float(component_strengths["z"][excited_id, ground_id]),
                "strength_sigma_plus": float(
                    component_strengths["sigma_plus"][excited_id, ground_id]
                ),
                "strength_sigma_minus": float(
                    component_strengths["sigma_minus"][excited_id, ground_id]
                ),
                "nphotons": nphotons,
                "branching": branching,
                "ground_largest_current": x_largest_current[ground_id],
                "ground_mJ_dominant": int(x_mj_dominant[ground_id]),
                "excited_largest_current": b_largest_current[excited_id],
            }
        )
    return records


def build_transition_grid(
    ez_values: Sequence[float] = DEFAULT_EZ_VALUES,
    ground_js: Sequence[int] = tuple(range(13)),
    excited_js: Sequence[int] = tuple(range(1, 11)),
    b_field: npt.NDArray[np.float64] = DEFAULT_B_FIELD,
    excited_j_padding: int = 2,
    coupling_cutoff: float = DEFAULT_COUPLING_CUTOFF,
    progress: bool = True,
    progress_reporter: Callable[[str], None] = print,
    timer: Callable[[], float] = time.perf_counter,
    workers: int = 1,
    tracking_workers: int = 1,
) -> TransitionGrid:
    workers = max(1, int(workers))
    tracking_workers = max(1, int(tracking_workers))
    ez_array = np.asarray(ez_values, dtype=np.float64)
    t_start = timer()
    t0 = timer()
    x_uncoupled, x_basis, x_transform, b_basis, b_label_basis, x_matrices, b_matrices = (
        _build_hamiltonian_sequences(
            ez_array, b_field, ground_js, excited_js, excited_j_padding
        )
    )
    build_matrices_s = timer() - t0

    t0 = timer()
    x_energies, x_eigenvectors, x_overlaps = _build_tracked_manifold(
        x_matrices,
        workers=tracking_workers,
    )
    b_energies, b_eigenvectors, b_overlaps = _build_tracked_manifold(
        b_matrices,
        workers=tracking_workers,
    )
    track_s = timer() - t0

    x_zero_labels, x_state_ids, x_zero_assignment_overlaps = _assign_labels_by_overlap(
        x_eigenvectors[0], x_basis, x_basis
    )
    b_zero_labels, b_state_ids, b_zero_assignment_overlaps = _assign_labels_by_overlap(
        b_eigenvectors[0], b_label_basis, b_basis
    )
    x_energies = x_energies[:, x_state_ids]
    b_energies = b_energies[:, b_state_ids]
    x_eigenvectors = x_eigenvectors[:, :, x_state_ids]
    b_eigenvectors = b_eigenvectors[:, :, b_state_ids]
    x_overlaps = x_overlaps[:, x_state_ids] if x_overlaps.size else x_overlaps
    b_overlaps = b_overlaps[:, b_state_ids] if b_overlaps.size else b_overlaps
    ground_js_by_label = np.array([label.J for label in x_zero_labels], dtype=int)
    candidates = _build_transition_candidates(x_zero_labels, b_zero_labels)
    x_basis_state_strings = _state_strings(x_basis)
    b_label_basis_state_strings = _state_strings(b_label_basis)

    t0 = timer()
    e1_operators = _build_e1_operator_matrices(x_basis, b_basis, workers=workers)
    e1_operator_s = timer() - t0
    setup_s = timer() - t_start
    if progress:
        progress_reporter(
            f"Setup: hamiltonians={build_matrices_s:.1f}s, "
            f"tracking={track_s:.1f}s (workers={tracking_workers}), "
            f"e1={e1_operator_s:.1f}s (workers={workers}), "
            f"total={setup_s:.1f}s"
        )

    slices: dict[float, TransitionGridSlice] = {}
    completed_slice_s = 0.0
    for idx, ez_v_cm in enumerate(ez_array):
        slice_start = timer()
        t0 = timer()
        coupling_matrices = _coupling_matrices_for_field(
            e1_operators,
            x_eigenvectors[idx],
            b_eigenvectors[idx],
        )
        component_strengths = _polarization_strength_matrices(coupling_matrices)
        strength_matrix = component_strengths["all"]
        matrix_coupling_s = timer() - t0

        t0 = timer()
        records = _line_records_for_slice(
            float(ez_v_cm),
            x_energies[idx],
            b_energies[idx],
            strength_matrix,
            component_strengths,
            candidates,
            ground_js_by_label,
            _state_strings_by_overlap(
                x_eigenvectors[idx],
                x_basis,
                x_basis,
                label_strings=x_basis_state_strings,
            ),
            _dominant_uncoupled_mj_values(
                x_eigenvectors[idx], x_transform, x_uncoupled
            ),
            _state_strings_by_overlap(
                b_eigenvectors[idx],
                b_label_basis,
                b_basis,
                label_strings=b_label_basis_state_strings,
            ),
            coupling_cutoff,
        )
        lines_s = timer() - t0
        lines = pd.DataFrame.from_records(records)
        if not lines.empty:
            lines = lines.sort_values(
                ["frequency_ir_mhz", "transition_name"]
            ).reset_index(drop=True)

        diagnostics = {
            "line_count": int(len(lines)),
            "candidate_count": int(len(candidates)),
            "min_x_tracking_overlap": (
                float(np.min(x_overlaps[idx - 1]))
                if idx > 0 and x_overlaps.size
                else 1.0
            ),
            "min_b_tracking_overlap": (
                float(np.min(b_overlaps[idx - 1]))
                if idx > 0 and b_overlaps.size
                else 1.0
            ),
            "min_x_zero_label_overlap": float(np.min(x_zero_assignment_overlaps)),
            "min_b_zero_label_overlap": float(np.min(b_zero_assignment_overlaps)),
        }
        slice_total_s = timer() - slice_start
        timings = {
            "matrix_coupling_s": matrix_coupling_s,
            "line_generation_s": lines_s,
            "slice_total_s": slice_total_s,
        }
        completed_slice_s += slice_total_s
        slices[float(ez_v_cm)] = TransitionGridSlice(
            ez_v_cm=float(ez_v_cm),
            lines=lines,
            timings=timings,
            diagnostics=diagnostics,
        )
        if progress:
            elapsed = timer() - t_start
            remaining = (
                (len(ez_array) - idx - 1) * completed_slice_s / (idx + 1)
            )
            progress_reporter(
                f"Ez={ez_v_cm:g} V/cm: {len(lines)} lines, "
                f"slice={timings['slice_total_s']:.1f}s, "
                f"elapsed={elapsed:.1f}s, eta={remaining:.1f}s"
            )

    metadata = {
        "schema_version": 6,
        "ground_js": list(map(int, ground_js)),
        "excited_js": list(map(int, excited_js)),
        "b_field_gauss": [float(v) for v in b_field],
        "branches": ["P", "Q", "R"],
        "polarization": "all, X, Y, Z, sigma+ and sigma- strengths about lab z",
        "frequency_units": "IR MHz",
        "uv_to_ir_frequency_factor": UV_TO_IR_FREQUENCY,
        "excited_construction_j_padding": int(excited_j_padding),
        "build_matrices_s": build_matrices_s,
        "tracking_s": track_s,
        "e1_operator_s": e1_operator_s,
        "workers": workers,
        "tracking_workers": tracking_workers,
        "total_s": timer() - t_start,
    }
    return TransitionGrid(ez_values=ez_array, slices=slices, metadata=metadata)


def _mf_component_label(row: pd.Series) -> str:
    return f"{int(row['mF_ground'])} -> {int(row['mF_excited'])}"


def _format_signed_int(value: int) -> str:
    return f"+{value}" if value > 0 else str(value)


def _parent_parity_from_label(label: object) -> str:
    if not isinstance(label, str):
        return "?"
    match = PARENT_PARITY_PATTERN.search(label)
    if match is None:
        return "?"
    return match.group(1)


def _summarize_parent_labels(labels: Sequence[object]) -> str:
    unique_labels = sorted(
        {
            str(label)
            for label in labels
            if isinstance(label, str) and label and not pd.isna(label)
        }
    )
    return "\n".join(unique_labels)


def _mf_branch_breakdown(part: pd.DataFrame) -> pd.DataFrame:
    breakdown = (
        part.groupby(["mF_ground", "mF_excited"], sort=True)
        .agg(state_pairs=("strength", "size"), strength=("strength", "sum"))
        .reset_index()
    )
    breakdown["label"] = breakdown.apply(_mf_component_label, axis=1)
    total_strength = float(breakdown["strength"].sum())
    if total_strength > 0:
        breakdown["share"] = breakdown["strength"] / total_strength
    else:
        breakdown["share"] = 0.0
    return breakdown.sort_values(
        ["strength", "mF_ground", "mF_excited"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def _mf_dominant_summary(breakdown: pd.DataFrame, top_n: int = 3) -> str:
    if breakdown.empty:
        return ""

    parts: list[str] = []
    for row in breakdown.head(top_n).itertuples(index=False):
        share_pct = int(round(float(row.share) * 100))
        parts.append(f"{row.label} ({share_pct}%)")

    remaining = len(breakdown) - min(len(breakdown), top_n)
    if remaining > 0:
        parts.append(f"+{remaining} more")
    return ", ".join(parts)


def _mf_detail_summary(breakdown: pd.DataFrame) -> str:
    if breakdown.empty:
        return ""

    parts = []
    for row in breakdown.itertuples(index=False):
        share_pct = int(round(float(row.share) * 100))
        parts.append(f"{row.label} ({share_pct}%, x{int(row.state_pairs)})")
    return ", ".join(parts)


@overload
def cluster_transition_components(
    lines: pd.DataFrame,
    resolving_frequency_mhz: float = 2.5,
    include_cluster_lines: Literal[False] = False,
) -> pd.DataFrame: ...


@overload
def cluster_transition_components(
    lines: pd.DataFrame,
    resolving_frequency_mhz: float = 2.5,
    include_cluster_lines: Literal[True] = True,
) -> tuple[pd.DataFrame, pd.DataFrame]: ...


def cluster_transition_components(
    lines: pd.DataFrame,
    resolving_frequency_mhz: float = 2.5,
    include_cluster_lines: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    if lines.empty:
        if include_cluster_lines:
            return pd.DataFrame(), pd.DataFrame()
        return pd.DataFrame()

    clusters: list[dict[str, object]] = []
    cluster_id = 0
    lines_for_grouping = lines
    cluster_lines: pd.DataFrame | None = None
    group_columns = ["transition_name"]
    if "excited_parent_parity" in lines_for_grouping.columns:
        group_columns.append("excited_parent_parity")
    elif "excited_label" in lines_for_grouping.columns:
        lines_for_grouping = lines.copy()
        lines_for_grouping["excited_parent_parity"] = lines_for_grouping[
            "excited_label"
        ].map(_parent_parity_from_label)
        group_columns.append("excited_parent_parity")
    if include_cluster_lines:
        cluster_line_columns = [
            column
            for column in [
                "excited_label",
                "J_ground",
                "F1_ground",
                "F_ground",
                "mF_ground",
                "mF_excited",
                "ground_mJ_dominant",
                "strength",
                "nphotons",
            ]
            if column in lines_for_grouping.columns
        ]
        cluster_lines = lines_for_grouping[cluster_line_columns].copy()
        cluster_lines["cluster_id"] = -1

    for group_key, group in lines_for_grouping.groupby(
        group_columns, sort=True, dropna=False
    ):
        if isinstance(group_key, tuple):
            transition_name = str(group_key[0])
            excited_parent_parity = str(group_key[1]) if len(group_key) > 1 else "?"
        else:
            transition_name = str(group_key)
            excited_parent_parity = str(group.iloc[0].get("excited_parent_parity", "?"))
        group = group.sort_values("frequency_ir_mhz", kind="mergesort")
        frequencies = group["frequency_ir_mhz"].to_numpy(dtype=float)
        split_points = (
            np.flatnonzero(np.diff(frequencies) > resolving_frequency_mhz) + 1
        )
        start_indices = np.concatenate(([0], split_points))
        stop_indices = np.concatenate((split_points, [len(group)]))

        for start, stop in zip(start_indices, stop_indices):
            part = group.iloc[start:stop]
            if include_cluster_lines and cluster_lines is not None:
                cluster_lines.loc[part.index, "cluster_id"] = cluster_id
            weights = part["strength"].to_numpy(dtype=float)
            part_frequencies = part["frequency_ir_mhz"].to_numpy(dtype=float)
            if np.sum(weights) > 0:
                center = float(np.average(part_frequencies, weights=weights))
                nphotons = float(
                    np.average(part["nphotons"].to_numpy(dtype=float), weights=weights)
                )
            else:
                center = float(np.mean(part_frequencies))
                nphotons = float(np.mean(part["nphotons"].to_numpy(dtype=float)))
            mf_pairs = part[["mF_ground", "mF_excited"]].drop_duplicates()
            delta_mf = np.unique(
                part["mF_excited"].to_numpy(dtype=int)
                - part["mF_ground"].to_numpy(dtype=int)
            )
            clusters.append(
                {
                    "cluster_id": cluster_id,
                    "transition_name": transition_name,
                    "branch": str(part.iloc[0]["branch"]),
                    "J_ground": int(part.iloc[0]["J_ground"]),
                    "excited_parent_parity": excited_parent_parity,
                    "frequency_ir_mhz": center,
                    "spread_ir_mhz": float(
                        np.max(part_frequencies) - np.min(part_frequencies)
                    ),
                    "strength": float(np.sum(weights)),
                    "components": int(len(part)),
                    "mf_branches": int(len(mf_pairs)),
                    "delta_mf": ", ".join(
                        _format_signed_int(int(value)) for value in delta_mf
                    ),
                    "nphotons": nphotons,
                }
            )
            cluster_id += 1
    clusters_df = (
        pd.DataFrame.from_records(clusters)
        .sort_values("frequency_ir_mhz")
        .reset_index(drop=True)
    )
    if not include_cluster_lines:
        return clusters_df
    if cluster_lines is None:
        return clusters_df, pd.DataFrame()
    return clusters_df, cluster_lines.reset_index(drop=True)


def transition_centroid(clusters: pd.DataFrame, transition_name: str) -> float:
    selected = clusters[clusters["transition_name"] == transition_name]
    if selected.empty:
        raise ValueError(f"Transition {transition_name!r} not found in grid slice.")
    weights = selected["strength"].to_numpy(dtype=float)
    frequencies = selected["frequency_ir_mhz"].to_numpy(dtype=float)
    if np.sum(weights) > 0:
        return float(np.average(frequencies, weights=weights))
    return float(np.mean(frequencies))


def select_clusters_for_display(
    clusters: pd.DataFrame,
    transition_name: str,
    energy_lim: tuple[float, float] = (-300.0, 300.0),
    ir_uv: str = "IR",
    calibration_offset_ir_mhz: float | None = None,
    reference_axis_shift_ir_mhz: float = 0.0,
    zero_field_clusters: pd.DataFrame | None = None,
) -> pd.DataFrame:
    convert = 1 if ir_uv == "IR" else 4
    offset = transition_centroid(clusters, transition_name)
    delta_column = f"Δ freq [{ir_uv}, MHz]"
    zero_field_delta_column = f"Δ from 0 V/cm [{ir_uv}, MHz]"

    selected = clusters.copy()
    selected[delta_column] = (selected["frequency_ir_mhz"] - offset) * convert
    if zero_field_clusters is not None and not zero_field_clusters.empty:
        references = {
            str(name): transition_centroid(zero_field_clusters, str(name))
            for name in zero_field_clusters["transition_name"].unique()
        }
        selected[zero_field_delta_column] = (
            selected["frequency_ir_mhz"]
            - selected["transition_name"].map(references)
        ) * convert
    if calibration_offset_ir_mhz is not None:
        selected[f"frequency [{ir_uv}, GHz]"] = (
            (
                selected["frequency_ir_mhz"]
                + calibration_offset_ir_mhz
                + float(reference_axis_shift_ir_mhz)
            )
            * convert
            / 1e3
        )
    return selected[
        (selected[delta_column] >= energy_lim[0])
        & (selected[delta_column] <= energy_lim[1])
    ].copy()


def enrich_cluster_display_fields(
    clusters: pd.DataFrame,
    cluster_lines: pd.DataFrame,
    include_full_detail: bool = False,
) -> pd.DataFrame:
    if (
        clusters.empty
        or cluster_lines.empty
        or "cluster_id" not in cluster_lines.columns
    ):
        return clusters.copy()

    selected_ids = clusters["cluster_id"].to_numpy(dtype=int)
    relevant_lines = cluster_lines[cluster_lines["cluster_id"].isin(selected_ids)]
    if relevant_lines.empty:
        return clusters.copy()

    summary_records: list[dict[str, object]] = []
    for cluster_id, part in relevant_lines.groupby("cluster_id", sort=False):
        breakdown = _mf_branch_breakdown(part)
        record: dict[str, object] = {
            "cluster_id": int(cluster_id),
            "mf_dominant": _mf_dominant_summary(breakdown),
        }
        if include_full_detail:
            labels = (
                part["excited_label"].tolist()
                if "excited_label" in part.columns
                else []
            )
            record["excited_parent_label"] = _summarize_parent_labels(labels)
            record["mf_detail"] = _mf_detail_summary(breakdown)
        summary_records.append(record)

    summary_df = pd.DataFrame.from_records(summary_records)
    return clusters.drop(
        columns=["excited_parent_label", "mf_dominant", "mf_detail", "mF"],
        errors="ignore",
    ).merge(summary_df, on="cluster_id", how="left")


def format_cluster_dataframe(selected: pd.DataFrame, ir_uv: str = "IR") -> pd.DataFrame:
    selected = selected.rename(
        columns={
            "transition_name": "transition",
            "excited_parent_parity": "P'",
            "excited_parent_label": "excited parent",
            "spread_ir_mhz": "spread [IR, MHz]",
            "strength": "strength",
            "components": "state pairs",
            "mf_branches": "mF branches",
            "delta_mf": "ΔmF",
            "mf_dominant": "dominant mF",
            "mf_detail": "mF detail",
            "nphotons": "photons",
        }
    )
    columns = [
        "transition",
        f"Δ freq [{ir_uv}, MHz]",
    ]
    if "cluster_id" in selected.columns:
        columns.insert(1, "cluster_id")
    if f"Δ from 0 V/cm [{ir_uv}, MHz]" in selected.columns:
        columns.append(f"Δ from 0 V/cm [{ir_uv}, MHz]")
    if f"frequency [{ir_uv}, GHz]" in selected.columns:
        columns.append(f"frequency [{ir_uv}, GHz]")
    columns.extend(
        column
        for column in [
            "P'",
            "spread [IR, MHz]",
            "state pairs",
            "mF branches",
            "ΔmF",
            "dominant mF",
            "strength",
            "photons",
            "excited parent",
            "mF detail",
        ]
        if column in selected.columns
    )
    return selected[columns].set_index("transition")


def generate_cluster_dataframe(
    clusters: pd.DataFrame,
    transition_name: str,
    energy_lim: tuple[float, float] = (-300.0, 300.0),
    ir_uv: str = "IR",
    calibration_offset_ir_mhz: float | None = None,
    reference_axis_shift_ir_mhz: float = 0.0,
    zero_field_clusters: pd.DataFrame | None = None,
    cluster_lines: pd.DataFrame | None = None,
    include_full_detail: bool = True,
) -> pd.DataFrame:
    selected = select_clusters_for_display(
        clusters,
        transition_name,
        energy_lim,
        ir_uv,
        calibration_offset_ir_mhz=calibration_offset_ir_mhz,
        reference_axis_shift_ir_mhz=reference_axis_shift_ir_mhz,
        zero_field_clusters=zero_field_clusters,
    )
    if cluster_lines is not None:
        selected = enrich_cluster_display_fields(
            selected,
            cluster_lines,
            include_full_detail=include_full_detail,
        )
    return format_cluster_dataframe(selected, ir_uv)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Precompute tracked transition grid.")
    parser.add_argument("--output", default=GRID_ARTIFACT)
    parser.add_argument("--ez-min", type=float, default=0.0)
    parser.add_argument("--ez-max", type=float, default=500.0)
    parser.add_argument("--ez-step", type=float, default=25.0)
    parser.add_argument("--ground-j-max", type=int, default=12)
    parser.add_argument("--excited-j-max", type=int, default=10)
    parser.add_argument("--excited-j-padding", type=int, default=2)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Default worker process count for parallel stages.",
    )
    parser.add_argument(
        "--e1-workers",
        type=int,
        default=None,
        help="Override worker process count for E1 operator construction.",
    )
    parser.add_argument(
        "--tracking-workers",
        type=int,
        default=None,
        help="Override worker process count for block eigenshuffle tracking.",
    )
    parser.add_argument(
        "--no-spinner",
        action="store_true",
        help="Disable the interactive elapsed-time spinner.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    e1_workers = args.workers if args.e1_workers is None else args.e1_workers
    tracking_workers = (
        args.workers if args.tracking_workers is None else args.tracking_workers
    )
    if e1_workers < 1:
        parser.error("--e1-workers must be at least 1")
    if tracking_workers < 1:
        parser.error("--tracking-workers must be at least 1")
    ez_values = np.arange(args.ez_min, args.ez_max + args.ez_step / 2, args.ez_step)
    total_start = time.perf_counter()
    with ElapsedSpinner(
        "Building transition grid",
        enabled=not args.no_spinner,
    ) as spinner:
        grid = build_transition_grid(
            ez_values=ez_values,
            ground_js=tuple(range(args.ground_j_max + 1)),
            excited_js=tuple(range(1, args.excited_j_max + 1)),
            excited_j_padding=args.excited_j_padding,
            workers=e1_workers,
            tracking_workers=tracking_workers,
            progress_reporter=spinner.print_line,
        )
        spinner.message = "Saving transition grid"
        save_transition_grid(grid, args.output)
    print(f"Done in {format_elapsed(time.perf_counter() - total_start)}. Wrote {args.output}")


if __name__ == "__main__":
    main()
