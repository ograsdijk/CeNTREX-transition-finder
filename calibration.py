from dataclasses import dataclass
from typing import Sequence

from centrex_tlf import transitions

from generate_transitions import Transition


@dataclass
class CalibrationTransition:
    transition: transitions.OpticalTransition
    frequency: float  # Frequency in IR MHz
    cesium_frequency: float  # Cesium Frequency in IR MHz


Q2_F1_5_2_F_3 = CalibrationTransition(
    transition=transitions.OpticalTransition(
        transitions.OpticalTransitionType.Q, J_ground=2, F1_excited=5 / 2, F_excited=3
    ),
    frequency=275848738,
    cesium_frequency=351730614,
)

R0_F1_1_2_F_1 = CalibrationTransition(
    transition=transitions.OpticalTransition(
        transitions.OpticalTransitionType.R, J_ground=0, F1_excited=1 / 2, F_excited=1
    ),
    frequency=275848556.92,
    cesium_frequency=351730618.5313543,
)


def get_offset(
    sorted_transitions: Sequence[Transition], calibration: CalibrationTransition
) -> float:
    idx_calibration = next(
        (
            idx
            for idx, obj in enumerate(sorted_transitions)
            if obj.transition == calibration.transition
        ),
        None,  # default if not found
    )
    if idx_calibration is None:
        raise ValueError("Calibration transition not found in transitions list.")
    offset = (
        -sorted_transitions[idx_calibration].weighted_energy + calibration.frequency
    )
    return offset
