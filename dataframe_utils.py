from typing import Optional, Sequence

import numpy as np
import pandas as pd
from centrex_tlf import transitions

from generate_transitions import Transition


def generate_dataframe_transitions(
    transition_interest: transitions.OpticalTransition,
    sorted_transitions: Sequence[Transition],
    energy_lim: tuple[float, float] = (-300, 300),
    ir_uv: str = "IR",
    calibration: Optional[float] = None,
) -> pd.DataFrame:
    if ir_uv == "IR":
        convert = 1
    else:
        convert = 4

    idx_center = next(
        (
            idx
            for idx, obj in enumerate(sorted_transitions)
            if obj.transition == transition_interest
        ),
        None,  # default if not found
    )
    if idx_center is None:
        raise ValueError("Transition of interest not found in transitions list.")

    offset = sorted_transitions[idx_center].weighted_energy
    energies = np.array([trans.weighted_energy for trans in sorted_transitions])

    indices_select = np.nonzero(
        ((energies - offset) * convert >= energy_lim[0])
        & ((energies - offset) * convert <= energy_lim[1])
    )[0]

    selected_transitions: list[Transition] = [
        sorted_transitions[i] for i in indices_select
    ]
    selected_energies = energies[indices_select]

    df = pd.DataFrame(
        {
            "transition": [trans.transition.name for trans in selected_transitions],
            f"Δ frequency [{ir_uv}, MHz]": (selected_energies - offset) * convert,
        }
    )
    if calibration is not None:
        df[f"frequency [{ir_uv}, GHz]"] = (
            (df[f"Δ frequency [{ir_uv}, MHz]"] + offset + calibration) * convert / 1e3
        )
    df["photons"] = [trans.nphotons for trans in selected_transitions]
    return df.set_index("transition")


def generate_dataframe_branching(
    transition_interest: transitions.OpticalTransition,
    sorted_transitions: Sequence[Transition],
    energy_lim: tuple[float, float] = (-300, 300),
    ir_uv: str = "IR",
) -> pd.DataFrame:
    if ir_uv == "IR":
        convert = 1
    else:
        convert = 4

    idx_center = next(
        (
            idx
            for idx, obj in enumerate(sorted_transitions)
            if obj.transition == transition_interest
        ),
        None,  # default if not found
    )
    if idx_center is None:
        raise ValueError("Transition of interest not found in transitions list.")

    offset = sorted_transitions[idx_center].weighted_energy
    energies = np.array([trans.weighted_energy for trans in sorted_transitions])

    indices_select = np.nonzero(
        ((energies - offset) * convert >= energy_lim[0])
        & ((energies - offset) * convert <= energy_lim[1])
    )[0]

    selected_transitions: list[Transition] = [
        sorted_transitions[i] for i in indices_select
    ]

    Js = np.unique([list(trans.branching.keys()) for trans in selected_transitions])

    branching = np.zeros((len(selected_transitions), Js.size))
    for idt, trans in enumerate(selected_transitions):
        br = np.zeros(Js.shape)
        dat = [(i, J) for i, J in enumerate(trans.branching.keys())]
        for i, Ji in dat:
            if trans.branching is not None:
                br[np.where(Js == Ji)] = trans.branching[Ji]
            else:
                br[np.where(Js == Ji)] = np.nan

        branching[idt] = br

    df = pd.DataFrame(
        {
            "transition": [trans.transition.name for trans in selected_transitions],
        }
    )
    for idj, Ji in enumerate(Js):
        if np.any(branching[:, idj] > 1e-4):
            df[f"J = {Ji}"] = branching[:, idj]

    return df.set_index("transition")
