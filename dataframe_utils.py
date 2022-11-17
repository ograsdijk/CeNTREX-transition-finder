from typing import Optional

import numpy as np
import pandas as pd
from centrex_tlf_hamiltonian import transitions

from hamiltonian_utils import SortedTransitions


def generate_dataframe(
    transitions_interest: list[transitions.OpticalTransition],
    sorted_transitions: SortedTransitions,
    energy_lim: tuple[float, float] = (-300, 300),
    ir_uv: str = "IR",
    calibration: Optional[float] = None,
) -> pd.DataFrame:

    if ir_uv == "IR":
        convert = 1
    else:
        convert = 4

    idx_center = np.where(sorted_transitions.transitions == transitions_interest)[0]
    offset = sorted_transitions.energies[idx_center]

    mask = ((sorted_transitions.energies - offset) * convert >= energy_lim[0]) & (
        (sorted_transitions.energies - offset) * convert <= energy_lim[1]
    )
    df = pd.DataFrame(
        {
            "transition": [t.name for t in sorted_transitions.transitions[mask]],
            f"Δ frequency [{ir_uv}, MHz]": (sorted_transitions.energies[mask] - offset)
            * convert,
        }
    )
    if calibration is not None:
        df[f"frequency [{ir_uv}, GHz]"] = (
            (df[f"Δ frequency [{ir_uv}, MHz]"] + offset + calibration) * convert / 1e3
        )
    return df.set_index("transition")
