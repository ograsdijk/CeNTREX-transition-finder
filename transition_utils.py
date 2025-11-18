import fractions
import re

from centrex_tlf import transitions


def format_transition_name(transition_name: str) -> str:
    """Format transition name to show F' as integer instead of float."""
    # Replace F'=X.0 with F'=X (where X is an integer)
    transition_name = re.sub(r"F'=(\d+)\.0\b", r"F'=\1", transition_name)
    return transition_name


def parse_transition(transition_name: str) -> transitions.OpticalTransition:
    t = transitions.OpticalTransitionType[transition_name[0]]
    J_ground = int(re.match(r"[A-Z]\((.*?)\).*?", transition_name).groups()[0])
    F1_str = re.match(r".*?F1'=(.*?) F'=.*?", transition_name).groups()[0]
    F1 = float(fractions.Fraction(F1_str))
    F = int(float(transition_name.split("=")[-1]))
    return transitions.OpticalTransition(t, J_ground, F1_excited=F1, F_excited=F)

