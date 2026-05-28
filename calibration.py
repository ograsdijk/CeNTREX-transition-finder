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
