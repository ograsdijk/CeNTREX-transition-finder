import itertools
from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt
import streamlit as st
from centrex_tlf_hamiltonian import hamiltonian, states, transitions
import centrex_tlf_couplings as couplings


def get_transition_from_state(
    ground_state: states.CoupledBasisState, excited_state: states.CoupledBasisState
) -> Optional[transitions.OpticalTransition]:
    if couplings.utils.check_transition_coupled_allowed(ground_state, excited_state):
        ΔJ = excited_state.J - ground_state.J
        if np.abs(ΔJ) > 1:
            return None
        return transitions.OpticalTransition(
            transitions.OpticalTransitionType(ΔJ),
            ground_state.J,
            excited_state.F1,
            excited_state.F,
        )


@st.experimental_memo
def generate_hamiltonian(
    J_ground: list[int] = [0, 1, 2, 3], J_excited: list[int] = [1, 2]
):
    ground_select = states.QuantumSelector(J=J_ground)
    excited_select = states.QuantumSelector(J=J_excited, P=[-1, 1])

    QN_X = states.generate_coupled_states_X(ground_select)
    QN_B = states.generate_coupled_states_B(excited_select)

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

    reduced_hamiltonian = hamiltonian.generate_total_reduced_hamiltonian(QN_X, QN_B)
    return reduced_hamiltonian


@dataclass
class Transition:
    indices_ground: list[int]
    indices_excited: list[int]
    transitions: list[transitions.OpticalTransition]


def get_transitions_from_hamiltonian(
    QN_X: list[states.CoupledBasisState], QN_B: list[states.CoupledBasisState]
) -> Transition:
    _transitions: list[transitions.OpticalTransition] = []
    indices_ground: list[int] = []
    indices_excited: list[int] = []
    for (idg, ground_state), (ide, excited_state) in itertools.product(
        enumerate(QN_X), enumerate(QN_B)
    ):
        if (
            trans := get_transition_from_state(ground_state, excited_state)
        ) is not None:
            if trans not in _transitions:
                _transitions.append(trans)
                indices_ground.append(idg)
                indices_excited.append(ide)
    return Transition(indices_ground, indices_excited, _transitions)


def get_energies(
    indices_ground: list[int],
    indices_excited: list[int],
    H_X: npt.NDArray[np.float64],
    H_B: npt.NDArray[np.float64],
    convert_IR=True,
) -> list[float]:
    if convert_IR:
        convert = 1 / 4
    else:
        convert = 1

    energies: list[float] = []
    for idg, ide in zip(indices_ground, indices_excited):
        energies.append((H_B[ide] - H_X[idg]) * convert)
    return energies


@dataclass
class SortedTransitions:
    indices_ground: npt.NDArray[np.int64]
    indices_excited: npt.NDArray[np.int64]
    energies: npt.NDArray[np.int64]
    transitions: npt.NDArray


def sort_transitions(
    transition: Transition,
    energies: list[float],
) -> SortedTransitions:
    indices_sort = np.argsort(energies)
    indices_ground = np.asarray(transition.indices_ground)
    indices_excited = np.asarray(transition.indices_excited)
    energies = np.asarray(energies)
    transition = np.asarray(transition.transitions)

    return SortedTransitions(
        indices_ground[indices_sort],
        indices_excited[indices_sort],
        energies[indices_sort],
        transition[indices_sort],
    )


def get_transitions(
    J_ground: list[int] = [0, 1, 2, 3], J_excited: list[int] = [1, 2]
) -> SortedTransitions:
    reduced_hamiltonian = generate_hamiltonian(J_ground, J_excited)

    possible_transitions = get_transitions_from_hamiltonian(
        [qn.largest for qn in reduced_hamiltonian.X_states],
        [qn.largest for qn in reduced_hamiltonian.B_states],
    )

    nr_ground_states = len(reduced_hamiltonian.X_states)
    nr_excited_states = len(reduced_hamiltonian.B_states)

    H_X = np.diag(reduced_hamiltonian.H_int)[:nr_ground_states].real / (
        2 * np.pi * 1e6
    )  # MHz
    H_B = np.diag(reduced_hamiltonian.H_int)[-nr_excited_states:].real / (
        2 * np.pi * 1e6
    )  # MHz

    energies = get_energies(
        possible_transitions.indices_ground,
        possible_transitions.indices_excited,
        H_X,
        H_B,
    )

    sorted_transitions = sort_transitions(possible_transitions, energies)
    return sorted_transitions
