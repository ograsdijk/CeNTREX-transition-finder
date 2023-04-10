import pickle
from pathlib import Path

import streamlit as st

from calibration import Q2_F1_5_2_F_3, get_offset
from dataframe_utils import generate_dataframe
from hamiltonian_utils import get_transitions
from plot_utils import generate_plot
from transition_utils import parse_transition

st.set_page_config(page_title="CeNTREX Transitions")


file_path = Path(__file__).parent.absolute()

if "sorted_transitions" not in st.session_state:
    pickled_files = [file.stem for file in file_path.glob("*.pkl")]
    if "sorted_transitions" in pickled_files:
        with open(file_path / "sorted_transitions.pkl", "rb") as f:
            sorted_transitions = pickle.load(f)
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

calibration = get_offset(sorted_transitions, Q2_F1_5_2_F_3)

with st.sidebar:
    st.title("Transition finder")
    transition_selector = st.selectbox(label="Transition", options=transition_names)
    col1, col2 = st.columns(2)
    with col1:
        energy_min = st.number_input(label="MHz", value=-300, max_value=0)
    with col2:
        energy_max = st.number_input(label="MHz", value=300, min_value=0)
    ir_uv = st.selectbox(label="UV or IR", options=["IR", "UV"], index=0)
    calculate_button = st.button("Calculate")


def generate_plot_dataframe():
    error = False
    if len(transition_selector) == 0:
        st.error("No transitions selected")
        error = True
    if energy_min == energy_max:
        st.error("Set valid energy span")
        error = True
    if error:
        return

    transitions_parsed = parse_transition(transition_selector)

    fig = generate_plot(
        transitions_parsed, sorted_transitions, (energy_min, energy_max), ir_uv
    )

    st.plotly_chart(fig, use_container_width=True)

    df = generate_dataframe(
        transitions_parsed,
        sorted_transitions,
        (energy_min, energy_max),
        ir_uv,
        calibration=calibration,
    )
    style = df.style.format(
        formatter={
            "Δ frequency [IR, MHz]": "{:.1f}",
            "frequency [IR, GHz]": "{:.3f}",
            "Δ frequency [UV, MHz]": "{:.1f}",
            "frequency [UV, GHz]": "{:.3f}",
        }
    )

    st.table(style)


if calculate_button:
    generate_plot_dataframe()
