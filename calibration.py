from dataclasses import dataclass

from centrex_tlf import transitions


@dataclass
class CalibrationTransition:
    transition: transitions.OpticalTransition
    frequency: float  # Frequency in IR MHz
    cesium_frequency: float  # Cesium Frequency in IR MHz


R0_F1_1_2_F_1 = CalibrationTransition(
    transition=transitions.OpticalTransition(
        transitions.OpticalTransitionType.R, J_ground=0, F1_excited=1 / 2, F_excited=1
    ),
    frequency=275848556.92,
    cesium_frequency=351730618.5313543,
)


R2_F1_7_2_F_3 = CalibrationTransition(
    transition=transitions.OpticalTransition(
        transitions.OpticalTransitionType.R, J_ground=2, F1_excited=7 / 2, F_excited=3
    ),
    frequency=275858770.677,
    cesium_frequency=351730547.000,
)


DEFAULT_CALIBRATION_TRANSITION = R2_F1_7_2_F_3
