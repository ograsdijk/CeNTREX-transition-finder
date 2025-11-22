import pickle

import numpy as np
from centrex_tlf import hamiltonian, states

from generate_transitions import (
    generate_transitions_E1,
    generate_transitions_E2,
    group_transitions,
    precalculate_decay_matrix,
)

E = np.array([0.0, 0.0, 0.0])  # Electric field in V/cm
B = np.array([0.0, 0.0, 1e-3])  # Magnetic field in Gauss

J_ground = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
J_excited = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

ground_select = states.QuantumSelector(J=J_ground)
excited_select = states.QuantumSelector(J=J_excited, P=[-1, 1])

QN_X = states.generate_coupled_states_X(ground_select)
QN_B = states.generate_coupled_states_B(excited_select, basis=states.Basis.CoupledP)

reduced_hamiltonian = hamiltonian.generate_total_reduced_hamiltonian(
    QN_X, QN_B, E=E, B=B
)

nr_ground_states = len(reduced_hamiltonian.X_states)
nr_excited_states = len(reduced_hamiltonian.B_states)

# User indicated reduced_hamiltonian is already diagonalized
QN_X_mixed = reduced_hamiltonian.X_states
QN_B_mixed = reduced_hamiltonian.B_states

H_X = np.diag(reduced_hamiltonian.H_int)[:nr_ground_states].real / (
    2 * np.pi * 1e6
)  # MHz
H_B = np.diag(reduced_hamiltonian.H_int)[-nr_excited_states:].real / (
    2 * np.pi * 1e6
)  # MHz

transitions_E1 = generate_transitions_E1(QN_X_mixed, QN_B_mixed)
transitions_E2 = generate_transitions_E2(QN_X_mixed, QN_B_mixed)

coupling_matrix = precalculate_decay_matrix(QN_X_mixed, QN_B_mixed)
ground_Js = np.array([s.largest.J for s in QN_X_mixed])

transitions_list_E1 = group_transitions(
    transitions_E1, H_X, H_B, coupling_matrix, ground_Js, QN_X_mixed, QN_B_mixed
)
transitions_list_E2 = group_transitions(
    transitions_E2, H_X, H_B, coupling_matrix, ground_Js, QN_X_mixed, QN_B_mixed
)

sorted_transitions = sorted(
    transitions_list_E1 + transitions_list_E2, key=lambda t: t.weighted_energy
)

with open("sorted_transitions.pkl", "wb") as f:
    pickle.dump(sorted_transitions, f)
