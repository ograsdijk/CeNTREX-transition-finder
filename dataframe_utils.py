from typing import Optional

import numpy as np
import pandas as pd
from centrex_tlf import transitions

from hamiltonian_utils import Transitions, unique_unsorted
from transition_utils import format_transition_name


def generate_dataframe_transitions(
    transitions_interest: transitions.OpticalTransition,
    sorted_transitions: Transitions,
    energy_lim: tuple[float, float] = (-300, 300),
    ir_uv: str = "IR",
    calibration: Optional[float] = None,
) -> pd.DataFrame:
    if ir_uv == "IR":
        convert = 1
    else:
        convert = 4

    indices_center = np.where(sorted_transitions.transitions == transitions_interest)[0]

    offset = sorted_transitions.energies_mean[indices_center]

    mask = ((sorted_transitions.energies_mean - offset) * convert >= energy_lim[0]) & (
        (sorted_transitions.energies_mean - offset) * convert <= energy_lim[1]
    )

    energies = sorted_transitions.energies_mean[mask]

    df = pd.DataFrame(
        {
            "transition": [
                format_transition_name(trans.name) for trans in sorted_transitions.transitions[mask]
            ],
            f"Δ frequency [{ir_uv}, MHz]": (energies - offset) * convert,
        }
    )
    if calibration is not None:
        df[f"frequency [{ir_uv}, GHz]"] = (
            (df[f"Δ frequency [{ir_uv}, MHz]"] + offset + calibration) * convert / 1e3
        )
    df["photons"] = [
        trans.photons for trans in sorted_transitions.transitions_data[mask]
    ]
    return df.set_index("transition")


def generate_dataframe_branching(
    transitions_interest: transitions.OpticalTransition,
    sorted_transitions: Transitions,
    energy_lim: tuple[float, float] = (-300, 300),
    ir_uv: str = "IR",
) -> pd.DataFrame:
    if ir_uv == "IR":
        convert = 1
    else:
        convert = 4

    indices_center = np.where(sorted_transitions.transitions == transitions_interest)[0]

    offset = sorted_transitions.energies_mean[indices_center]

    mask = ((sorted_transitions.energies_mean - offset) * convert >= energy_lim[0]) & (
        (sorted_transitions.energies_mean - offset) * convert <= energy_lim[1]
    )

    Js = np.unique(
        [
            item
            for row in [
                [int(trans.br.iloc[i].name[-2]) for i in range(len(trans.br))]
                for trans in sorted_transitions.transitions_data[mask]
            ]
            for item in row
        ]
    )

    branching = np.zeros((mask.sum(), Js.size))
    for idt, trans in enumerate(sorted_transitions.transitions_data[mask]):
        br = np.zeros(Js.shape)
        dat = [(i, int(trans.br.iloc[i].name[-2])) for i in range(len(trans.br))]
        for i, Ji in dat:
            br[np.where(Js == Ji)] = trans.br.iloc[i].values
        branching[idt] = br
    df = pd.DataFrame(
        {
            "transition": [
                format_transition_name(trans.name) for trans in sorted_transitions.transitions[mask]
            ],
        }
    )
    for idj, Ji in enumerate(Js):
        if np.any(branching[:, idj] > 1e-4):
            df[f"J = {Ji}"] = branching[:, idj]

    return df.set_index("transition")
