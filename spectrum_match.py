import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from centrex_tlf import transitions

from calibration import Q2_F1_5_2_F_3, R0_F1_1_2_F_1, get_offset
from hamiltonian_utils import SortedTransitions, get_transitions
from spectrum_utils import find_overlap_searchsorted

st.set_page_config(page_title="CeNTREX Spectrum Matching")

file_path = Path(__file__).parent.absolute()

if "sorted_transitions" not in st.session_state:
    pickled_files = [file.stem for file in file_path.glob("*.pkl")]
    if "sorted_transitions" in pickled_files:
        with open(file_path / "sorted_transitions.pkl", "rb") as f:
            sorted_transitions: SortedTransitions = pickle.load(f)
    else:
        sorted_transitions = get_transitions(
            J_ground=[0, 1, 2, 3, 4, 5, 6, 7, 8], J_excited=[1, 2, 3, 4, 5, 6]
        )
        with open(file_path / "sorted_transitions.pkl", "wb") as f:
            pickle.dump(sorted_transitions, f)

    st.session_state["sorted_transitions"] = sorted_transitions
else:
    sorted_transitions = st.session_state["sorted_transitions"]

transition_names = [trans.name for trans in sorted_transitions.transitions]
transition_names.sort()

calibration = get_offset(sorted_transitions, R0_F1_1_2_F_1)

with st.sidebar:
    st.title("Spectrum matching")
    frequencies_measured = st.multiselect(
        label="Measured frequencies [MHz]", options=np.arange(-5000, 5000, 1)
    )
    frequency_resolve = st.number_input(
        label="Resolving frequency [MHz]", value=15, min_value=1
    )
    ir_uv = st.selectbox(label="UV or IR", options=["IR", "UV"], index=0)
    calculate_button = st.button("Calculate")


if calculate_button:

    if ir_uv == "UV":
        convert = 4
    else:
        convert = 1

    sorted_frequencies_measured = np.sort(frequencies_measured) * convert
    (
        indices_overlap,
        energies_overlap,
        frequencies_measured_resolved,
    ) = find_overlap_searchsorted(
        sorted_frequencies_measured, sorted_transitions.energies, frequency_resolve
    )

    def gaussian(x, mu, sig):
        return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))

    _energy = np.linspace(-15, 15, 201)
    lineshape = gaussian(_energy, 0, 2.5)

    fig = go.Figure()

    indices_overlap_flattened = [val for sublist in indices_overlap for val in sublist]
    idt_min = np.min(indices_overlap_flattened)
    idt_max = np.max(indices_overlap_flattened)

    mean_offset = np.mean(
        np.asarray(frequencies_measured_resolved)
        - (
            np.asarray(energies_overlap)
            - sorted_transitions.energies[idt_min]
            + frequencies_measured[0]
        )
    )

    # plot measured frequencies
    for idp, p in enumerate(frequencies_measured_resolved):
        fig.add_trace(
            go.Scatter(
                x=(_energy / 5 + p) * convert,
                y=lineshape,
                name="",
                meta=[f"peak {idp+1}"],
                hovertemplate="%{meta[0]}",
                line={"color": "black"},
                mode="lines",
                showlegend=False,
            )
        )

    # plot found peaks
    for idt in range(idt_min, idt_max + 1):
        trans = sorted_transitions.transitions[idt]
        loc = sorted_transitions.energies[idt]
        if trans.t == transitions.OpticalTransitionType.R:
            color = "#9467bd"
        elif trans.t == transitions.OpticalTransitionType.Q:
            color = "#ff7f0e"
        elif trans.t == transitions.OpticalTransitionType.P:
            color = "#2ca02c"
        fig.add_trace(
            go.Scatter(
                x=(
                    _energy
                    + loc
                    - sorted_transitions.energies[idt_min]
                    + frequencies_measured[0]
                    + mean_offset
                )
                * convert,
                y=lineshape,
                name=trans.name,
                line={"color": color},
                meta=[trans.name],
                hovertemplate="%{meta[0]}",
                mode="lines",
                showlegend=False,
            )
        )
    fig.update_layout(
        title="Spectral match",
        xaxis_title=f"frequency [IR, {ir_uv}]",
        font=dict(size=14)
        # font=dict(
        #     family="Courier New, monospace",
        #     size=18,
        #     color="RebeccaPurple"
        # )
    )

    st.plotly_chart(fig, use_container_width=True)

    # display the relative and absolute frequencies
    _peak = []
    _transitions = []
    _frequencies = []
    _frequencies_absolute = []
    for ido, idx in enumerate(indices_overlap):
        if isinstance(idx, list):
            for idy in idx:
                _transitions.append(sorted_transitions.transitions[idy].name)
                _frequencies.append(
                    (
                        sorted_transitions.energies[idy]
                        - sorted_transitions.energies[idt_min]
                        + frequencies_measured[0]
                        + mean_offset
                    )
                    * convert
                )
                _frequencies_absolute.append(
                    (sorted_transitions.energies[idy] + calibration)
                    * convert
                    / 1e3  # to GHz
                )
                _peak.append(ido + 1)

    df = pd.DataFrame(
        {
            "peak": _peak,
            "transition": _transitions,
            f"Δ frequency [{ir_uv}, MHz]": _frequencies,
            f"frequency [{ir_uv}, GHz]": _frequencies_absolute,
        }
    )
    df = df.set_index("transition")

    style = df.style.format(
        formatter={
            "Δ frequency [IR, MHz]": "{:.1f}",
            "frequency [IR, GHz]": "{:.3f}",
            "Δ frequency [UV, MHz]": "{:.1f}",
            "frequency [UV, GHz]": "{:.3f}",
        }
    )

    st.table(style)
