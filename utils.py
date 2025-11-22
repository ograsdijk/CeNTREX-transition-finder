import fractions
import re

import numpy as np
import numpy.typing as npt
from centrex_tlf import transitions


def parse_transition(transition_name: str) -> transitions.OpticalTransition:
    t = transitions.OpticalTransitionType[transition_name[0]]
    match = re.match(r"[A-Z]\((.*?)\).*?", transition_name)
    if match is None:
        raise ValueError("Could not parse J_ground from transition name.")
    J_ground = int(match.groups()[0])
    match = re.match(r".*?F1'=(.*?) F'=.*?", transition_name)
    if match is None:
        raise ValueError("Could not parse F1_excited from transition name.")
    F1_str = match.groups()[0]
    F1 = float(fractions.Fraction(F1_str))
    F = int(float(transition_name.split("=")[-1]))
    return transitions.OpticalTransition(t, J_ground, F1_excited=F1, F_excited=F)


def unique_unsorted(
    trans: npt.NDArray[np.object_],
) -> list[transitions.OpticalTransition]:
    unique: list[transitions.OpticalTransition] = []
    for t in trans:
        if t not in unique:
            unique.append(t)
    return unique
