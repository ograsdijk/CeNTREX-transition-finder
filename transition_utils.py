import fractions
import re

from centrex_tlf_hamiltonian import transitions


def parse_transition(transition_name: str) -> transitions.OpticalTransition:
    t = transitions.OpticalTransitionType[transition_name[0]]
    J_ground = int(re.match("[A-Z]\((.*?)\).*?", transition_name).groups()[0])
    F1_str = re.match(".*?F1'=(.*?) F'=.*?", transition_name).groups()[0]
    F1 = float(fractions.Fraction(F1_str))
    F = int(transition_name.split("=")[-1])
    return transitions.OpticalTransition(t, J_ground, F1=F1, F=F)
