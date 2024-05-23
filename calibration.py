from dataclasses import dataclass

import numpy as np
from centrex_tlf import transitions

from hamiltonian_utils import Transitions


@dataclass
class CalibrationTransition:
    transition: transitions.OpticalTransition
    frequency: float  # Frequency in IR MHz
    cesium_frequency: float  # Cesium Frequency in IR MHz


Q2_F1_5_2_F_3 = CalibrationTransition(
    transition=transitions.OpticalTransition(
        transitions.OpticalTransitionType.Q, J_ground=2, F1=5 / 2, F=3
    ),
    frequency=275848738,
    cesium_frequency=351730614,
)

R0_F1_1_2_F_1 = CalibrationTransition(
    transition=transitions.OpticalTransition(
        transitions.OpticalTransitionType.R, J_ground=0, F1=1 / 2, F=1
    ),
    frequency=275848556.92,
    cesium_frequency=351730618.5313543,
)


def get_offset(
    sorted_transitions: Transitions, calibration: CalibrationTransition
) -> float:
    # print(calibration.transition)
    idx_calibration = np.where(
        sorted_transitions.transitions == calibration.transition
    )[0][0]
    offset = -sorted_transitions.energies_mean[idx_calibration] + calibration.frequency
    return offset
