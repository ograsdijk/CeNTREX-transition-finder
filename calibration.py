from dataclasses import dataclass

import numpy as np
from centrex_tlf import transitions

from hamiltonian_utils import SortedTransitions


@dataclass
class CalibrationTransition:
    transition: transitions.OpticalTransition
    frequency: float  # Frequency in IR MHz


Q2_F1_5_2_F_3 = CalibrationTransition(
    transitions.OpticalTransition(
        transitions.OpticalTransitionType.Q, J_ground=2, F1=5 / 2, F=3
    ),
    275848720,
)

R0_F1_1_2_F_1 = CalibrationTransition(
    transitions.OpticalTransition(
        transitions.OpticalTransitionType.R, J_ground=0, F1=1 / 2, F=1
    ),
    275848554,
)


def get_offset(
    sorted_transitions: SortedTransitions, calibration: CalibrationTransition
) -> float:
    # print(calibration.transition)
    idx_calibration = np.where(
        sorted_transitions.transitions == calibration.transition
    )[0][0]
    offset = -sorted_transitions.energies[idx_calibration] + calibration.frequency
    return offset
