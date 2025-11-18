import itertools
from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import streamlit as st
from centrex_tlf import couplings, hamiltonian, states, transitions, utils


def get_transition_from_state(
    ground_state: states.CoupledBasisState, excited_state: states.CoupledBasisState
) -> Optional[transitions.OpticalTransition]:
    if couplings.utils.check_transition_coupled_allowed(ground_state, excited_state):
        ΔJ = excited_state.J - ground_state.J
        ΔJs = [val.value for val in transitions.OpticalTransitionType]
        if (ΔJ > max(ΔJs)) or (ΔJ < min(ΔJs)):
            return None
        return transitions.OpticalTransition(
            transitions.OpticalTransitionType(ΔJ),
            ground_state.J,
            excited_state.F1,
            excited_state.F,
        )


@st.cache_data
def generate_hamiltonian(
    J_ground: list[int] = None, J_excited: list[int] = None
):
    if J_ground is None:
        J_ground = [0, 1, 2, 3]
    if J_excited is None:
        J_excited = [1, 2]
    ground_select = states.QuantumSelector(J=J_ground)
    excited_select = states.QuantumSelector(J=J_excited, P=[-1, 1])

    QN_X = states.generate_coupled_states_X(ground_select)
    QN_B = states.generate_coupled_states_B(excited_select, basis=states.Basis.CoupledP)

    possible_transitions = []
    for (idg, ground_state), (ide, excited_state) in itertools.product(
        enumerate(QN_X), enumerate(QN_B)
    ):
        if (
            trans := get_transition_from_state(ground_state, excited_state)
        ) is not None:
            if trans not in possible_transitions:
                possible_transitions.append(trans)

    ground_select = [trans.qn_select_ground for trans in possible_transitions]
    excited_select = [trans.qn_select_excited for trans in possible_transitions]

    QN_X = states.generate_coupled_states_X(ground_select)
    QN_B = states.generate_coupled_states_B(excited_select)

    reduced_hamiltonian = hamiltonian.generate_total_reduced_hamiltonian(
        QN_X, QN_B, B=np.array([0, 0, 1e-3])
    )
    return reduced_hamiltonian


@dataclass
class Transition:
    transition: transitions.OpticalTransition
    energies: npt.NDArray[np.float64]
    states_ground: list[states.State]
    states_excited: list[states.State]
    coupling_elements: npt.NDArray[complex]
    photons: float
    br: pd.DataFrame


@dataclass
class Transitions:
    transitions: npt.NDArray
    transitions_data: npt.NDArray
    energies_mean: npt.NDArray


def get_transitions_from_hamiltonian(
    QN_X: list[states.CoupledBasisState],
    QN_B: list[states.CoupledBasisState],
    H_X: npt.NDArray[complex],
    H_B: npt.NDArray[complex],
) -> Transitions:
    _transitions: list[transitions.OpticalTransition] = []
    indices_ground: list[int] = []
    indices_excited: list[int] = []
    states_ground: list[states.State] = []
    states_excited: list[states.State] = []
    coupling_elements: list[complex] = []
    for (idg, ground_state), (ide, excited_state) in itertools.product(
        enumerate(QN_X), enumerate(QN_B)
    ):
        if (
            trans := get_transition_from_state(
                ground_state.largest, excited_state.largest
            )
        ) is not None:
            _transitions.append(trans)
            indices_ground.append(idg)
            indices_excited.append(ide)
            states_ground.append(QN_X[idg])
            states_excited.append(QN_B[ide])
            coupling_elements.append(
                couplings.calculate_ED_ME_mixed_state(
                    QN_X[idg],
                    QN_B[ide],
                    reduced=False,
                    normalize_pol=True,
                )
            )

    energies = get_energies(indices_ground, indices_excited, H_X, H_B)

    unique = unique_unsorted(_transitions)
    transitions_data = []
    energies_mean = []
    for trans in unique:
        mask = trans == np.array(_transitions)
        br = couplings.generate_br_dataframe(
            QN_X,
            np.array(unique_unsorted(np.array(states_excited)[mask])),
            group_ground="J",
        )
        Js = np.array([int(br.iloc[i].name[-2]) for i in range(len(br))])
        br_off = np.sum([br.iloc[i].values for i in np.where(Js != trans.J_ground)[0]])
        n_photons = 1 / br_off if 1 / br_off < 100 else 100

        transitions_data.append(
            Transition(
                trans,
                energies[mask],
                np.array(states_ground)[mask],
                np.array(states_excited)[mask],
                np.array(coupling_elements)[mask],
                n_photons,
                br,
            )
        )
        mask_nonzero = np.array(coupling_elements)[mask] != 0
        energies_mean.append(energies[mask][mask_nonzero].mean())

    return Transitions(
        np.array(unique), np.array(transitions_data), np.array(energies_mean)
    )


def get_energies(
    indices_ground: list[int],
    indices_excited: list[int],
    H_X: npt.NDArray[complex],
    H_B: npt.NDArray[complex],
    convert_IR=True,
) -> npt.NDArray[np.float64]:
    if convert_IR:
        convert = 1 / 4
    else:
        convert = 1

    energies: list[float] = []
    for idg, ide in zip(indices_ground, indices_excited):
        energies.append(((H_B[ide] - H_X[idg]) * convert).real)
    return np.array(energies)


def sort_transitions(
    transitions: Transitions,
) -> Transitions:
    indices_sort = np.argsort(transitions.energies_mean)
    return Transitions(
        transitions.transitions[indices_sort],
        transitions.transitions_data[indices_sort],
        transitions.energies_mean[indices_sort],
    )


def get_transitions(
    J_ground: list[int] = None, J_excited: list[int] = None
) -> Transitions:
    if J_ground is None:
        J_ground = [0, 1, 2, 3]
    if J_excited is None:
        J_excited = [1, 2]
    reduced_hamiltonian = generate_hamiltonian(J_ground, J_excited)

    nr_ground_states = len(reduced_hamiltonian.X_states)
    nr_excited_states = len(reduced_hamiltonian.B_states)

    H_X = np.diag(reduced_hamiltonian.H_int)[:nr_ground_states].real / (
        2 * np.pi * 1e6
    )  # MHz
    H_B = np.diag(reduced_hamiltonian.H_int)[-nr_excited_states:].real / (
        2 * np.pi * 1e6
    )  # MHz

    possible_transitions = get_transitions_from_hamiltonian(
        reduced_hamiltonian.X_states, reduced_hamiltonian.B_states, H_X, H_B
    )

    sorted_transitions = sort_transitions(possible_transitions)
    return sorted_transitions


def unique_unsorted(
    trans: npt.NDArray[np.object_]
) -> list[transitions.OpticalTransition]:
    unique: list[transitions.OpticalTransition] = []
    for t in trans:
        if t not in unique:
            unique.append(t)
    return unique
