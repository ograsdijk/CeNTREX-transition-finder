import itertools
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt
from centrex_tlf import couplings, hamiltonian, states, transitions


@dataclass
class Transition:
    """
    Data class representing a grouped optical transition between ground and excited states.
    """

    transition: transitions.OpticalTransition
    ground_states: list[states.CoupledState]
    excited_states: list[states.CoupledState]
    coupling_elements_E1: list[complex]
    coupling_elements_E2: list[complex]
    ground_energies: npt.NDArray[np.float64]
    excited_energies: npt.NDArray[np.float64]
    weighted_energy: float
    nphotons: float | None
    branching: dict[int, float] | None
    ground_states_participating: int

    def __repr__(self) -> str:
        if self.nphotons is not None:
            return f"Transition({self.transition.name}, nγ={self.nphotons:.2f})"
        else:
            return f"Transition({self.transition.name}, nγ=None)"


def get_transition_from_state(
    ground_state: states.CoupledBasisState,
    excited_state: states.CoupledBasisState,
    transition: str = "E1",
) -> transitions.OpticalTransition | None:
    """
    Determine the optical transition type between a ground and excited state.

    Args:
        ground_state (states.CoupledBasisState): The ground state.
        excited_state (states.CoupledBasisState): The excited state.
        transition (str, optional): The transition type ("E1" or "E2"). Defaults to "E1".

    Returns:
        transitions.OpticalTransition | None: The optical transition object if allowed, else None.
    """
    if transition == "E1":
        checker = couplings.utils.check_transition_coupled_allowed
    elif transition == "E2":
        checker = couplings.utils.check_transition_coupled_allowed_E2
    else:
        raise ValueError(f"{transition} coupling not supported")
    if checker(ground_state, excited_state):
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


def generate_transitions_E1(
    ground_states: Sequence[states.CoupledState],
    excited_states: Sequence[states.CoupledState],
) -> list[tuple[int, int, transitions.OpticalTransition, complex, complex]]:
    """
    Generate E1 transitions between ground and excited states.

    Args:
        ground_states (Sequence[states.CoupledState]): List of ground states.
        excited_states (Sequence[states.CoupledState]): List of excited states.

    Returns:
        list[tuple[int, int, transitions.OpticalTransition, complex, complex]]:
            List of tuples containing (ground_idx, excited_idx, transition, E1_coupling, E2_coupling).
            E2_coupling is always 0j for this function.
    """
    transitions_E1: list[
        tuple[int, int, transitions.OpticalTransition, complex, complex]
    ] = []
    for (idg, ground_state), (ide, excited_state) in itertools.product(
        enumerate(ground_states),
        enumerate(excited_states),
    ):
        if abs(excited_state.largest.J - ground_state.largest.J) > 1:
            continue
        if excited_state.largest.P == ground_state.largest.P:
            continue
        cpl = hamiltonian.generate_ED_ME_mixed_state(
            excited_state, ground_state, reduced=True
        )
        transition = get_transition_from_state(
            ground_state.largest, excited_state.largest
        )
        if transition is not None:
            transitions_E1.append((idg, ide, transition, cpl, 0j))
    return transitions_E1


def generate_transitions_E2(
    ground_states: Sequence[states.CoupledState],
    excited_states: Sequence[states.CoupledState],
) -> list[tuple[int, int, transitions.OpticalTransition, complex, complex]]:
    """
    Generate E2 transitions between ground and excited states.
    Also calculates E1 couplings for the same transitions due to mixing.

    Args:
        ground_states (Sequence[states.CoupledState]): List of ground states.
        excited_states (Sequence[states.CoupledState]): List of excited states.

    Returns:
        list[tuple[int, int, transitions.OpticalTransition, complex, complex]]:
            List of tuples containing (ground_idx, excited_idx, transition, E1_coupling, E2_coupling).
    """
    transitions_E2: list[
        tuple[int, int, transitions.OpticalTransition, complex, complex]
    ] = []
    for (idg, ground_state), (ide, excited_state) in itertools.product(
        enumerate(ground_states),
        enumerate(excited_states),
    ):
        if abs(excited_state.largest.J - ground_state.largest.J) != 2:
            continue
        if excited_state.largest.P != ground_state.largest.P:
            continue
        cpl_E2 = hamiltonian.generate_EQ_ME_mixed_state(
            excited_state, ground_state, reduced=True
        )
        cpl_E1 = hamiltonian.generate_ED_ME_mixed_state(
            excited_state, ground_state, reduced=True
        )
        transition = get_transition_from_state(
            ground_state.largest, excited_state.largest, "E2"
        )
        if (transition is not None) and (abs(cpl_E2) > 1e-14):
            transitions_E2.append((idg, ide, transition, cpl_E1, cpl_E2))
    return transitions_E2


def precalculate_decay_matrix(ground_states, excited_states) -> npt.NDArray[np.float64]:
    """
    Precalculate the decay matrix (sum of squared E1 couplings) for all pairs of states.

    Args:
        ground_states (list): List of ground states.
        excited_states (list): List of excited states.

    Returns:
        npt.NDArray[np.float64]: Matrix of decay strengths with shape (n_excited, n_ground).
    """
    n_X = len(ground_states)
    n_B = len(excited_states)
    decay_matrix = np.zeros((n_B, n_X), dtype=np.float64)

    pol_vecs = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        np.array([0, 0, 1], dtype=complex),
    ]

    for i, excited in enumerate(excited_states):
        for j, ground in enumerate(ground_states):
            if excited.largest.P == ground.largest.P:
                continue
            strength = 0.0
            for pol in pol_vecs:
                cpl = hamiltonian.generate_ED_ME_mixed_state(
                    excited, ground, pol_vec=pol, reduced=False
                )
                strength += np.abs(cpl) ** 2
            if strength < 1e-14:
                continue
            decay_matrix[i, j] = strength
    return decay_matrix


def group_transitions(
    transitions_sequence: list[
        tuple[int, int, transitions.OpticalTransition, complex, complex]
    ],
    H_X: npt.NDArray[np.float64],
    H_B: npt.NDArray[np.float64],
    coupling_matrix: npt.NDArray[np.float64],
    ground_Js: npt.NDArray[np.int_],
    ground_states: list[states.CoupledState],
    excited_states: list[states.CoupledState],
    mixing_cutoff: float = 0.1,
    E2_weight: float = 1e-6,
    vibrational_branching: float = 0.99,
    coupling_cutoff: float = 1e-3,
) -> list[Transition]:
    """
    Group transitions by their OpticalTransition type and calculate their properties
    including branching ratios, nphotons, and weighted energies.

    Args:
        transitions_sequence (list): List of transition tuples.
        H_X (npt.NDArray[np.float64]): Ground state energies.
        H_B (npt.NDArray[np.float64]): Excited state energies.
        coupling_matrix (npt.NDArray[np.float64]): Precalculated decay matrix.
        ground_Js (npt.NDArray[np.int_]): Array of J values for ground states.
        ground_states (list[states.State]): List of ground state objects.
        excited_states (list[states.State]): List of excited state objects.
        mixing_cutoff (float, optional): Cutoff for state mixing. Defaults to 0.1.
        E2_weight (float, optional): Weighting factor for E2 transitions. Defaults to 1e-6.
        vibrational_branching (float, optional): Vibrational branching ratio. Defaults to 0.99.
        coupling_cutoff (float, optional): Cutoff for E1 coupling to count participating ground states. Defaults to 1e-3.

    Returns:
        list[Transition]: List of grouped Transition objects.
    """
    # Grouping using a dictionary for O(N) performance
    grouped_data = {}

    for idg, ide, transition, cpl_E1, cpl_E2 in transitions_sequence:
        if transition not in grouped_data:
            grouped_data[transition] = {
                "ground_indices": [],
                "excited_indices": [],
                "coupling_elements_E1": [],
                "coupling_elements_E2": [],
            }

        data = grouped_data[transition]
        data["ground_indices"].append(idg)
        data["excited_indices"].append(ide)
        data["coupling_elements_E1"].append(cpl_E1)
        data["coupling_elements_E2"].append(cpl_E2)

    transitions_list: list[Transition] = []
    unique_Js = np.unique(ground_Js)

    for transition_type, data in grouped_data.items():
        g_indices = data["ground_indices"]
        e_indices = data["excited_indices"]
        cpls_E1 = data["coupling_elements_E1"]
        cpls_E2 = data["coupling_elements_E2"]

        # Check if all necessary ground states are present
        # Get all J's involved in the excited states
        excited_Js = set()
        for idx in e_indices:
            state = excited_states[idx]
            # state is a mixed state (states.State)
            # We want J of basis_state where amp is significant
            for amp, basis_state in state:
                if abs(amp) > mixing_cutoff:  # Threshold for mixing
                    excited_Js.add(basis_state.J)

        required_Js = set()
        for J in excited_Js:
            required_Js.update({J - 1, J, J + 1})
        required_Js = {J for J in required_Js if J >= 0}

        missing_Js = [J for J in required_Js if J not in unique_Js]

        branching = None
        nphotons = None

        if not missing_Js:
            # Branching calculation using pre-calculated matrix (E1 decays)
            # Get couplings for these excited states: shape (N_exc, N_ground)
            culs = coupling_matrix[e_indices, :]

            # Total decay rate per excited state
            total_decay = np.sum(culs, axis=1)

            # Avoid division by zero
            total_decay[total_decay == 0] = 1.0

            # Branching fractions matrix: (N_exc, N_ground)
            br_matrix = culs / total_decay[:, None]

            # Calculate branching for each J
            branching = {}
            for J in unique_Js:
                mask = ground_Js == J
                if np.any(mask):
                    br_vals = np.sum(br_matrix[:, mask], axis=1)
                    branching[int(J)] = float(np.mean(br_vals))

            # nphotons
            target_J = transition_type.J_ground
            nphotons = 1 / (1 - vibrational_branching * branching.get(target_J, 0))

        ground_energies = H_X[g_indices]
        excited_energies = H_B[e_indices]

        energies = excited_energies - ground_energies

        # Use stored coupling elements (which are now complex) to calculate weights
        weights = (
            np.abs(np.array(cpls_E1)) ** 2 + E2_weight * np.abs(np.array(cpls_E2)) ** 2
        )

        if np.sum(weights) != 0:
            weighted_energy = float(np.average(energies, weights=weights))
        else:
            weighted_energy = float(np.mean(energies))

        # Calculate ground_states_participating
        # Unique ground states with E1 coupling > coupling_cutoff
        participating_indices = {
            idx for idx, cpl in zip(g_indices, cpls_E1) if abs(cpl) > coupling_cutoff
        }
        ground_states_participating = len(participating_indices)

        transitions_list.append(
            Transition(
                transition=transition_type,
                ground_states=[ground_states[i] for i in g_indices],
                excited_states=[excited_states[i] for i in e_indices],
                coupling_elements_E1=cpls_E1,
                coupling_elements_E2=cpls_E2,
                ground_energies=ground_energies,
                excited_energies=excited_energies,
                weighted_energy=weighted_energy,
                nphotons=nphotons,
                branching=branching,
                ground_states_participating=ground_states_participating,
            )
        )

    return transitions_list
