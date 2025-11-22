import pickle
from pathlib import Path

import numpy as np
import numpy.typing as npt
import streamlit as st
from centrex_tlf import utils

st.set_page_config(page_title="CeNTREX Transitions")

from calibration import R0_F1_1_2_F_1, get_offset
from dataframe_utils import generate_dataframe_branching, generate_dataframe_transitions
from generate_transitions import Transition
from plot_utils import generate_plot
from utils import parse_transition

file_path = Path(__file__).parent.absolute()

if "sorted_transitions" not in st.session_state:
    pickled_files = [file.stem for file in file_path.glob("*.pkl")]
    if "sorted_transitions" in pickled_files:
        with open(file_path / "sorted_transitions.pkl", "rb") as f:
            sorted_transitions: list[Transition] = pickle.load(f)
        print("opened file")
    else:
        raise RuntimeError(
            "No precomputed transitions found. Please run 'compute_transitions.py' to generate the required data."
        )
    st.session_state["sorted_transitions"] = sorted_transitions
else:
    sorted_transitions = st.session_state["sorted_transitions"]

transition_names = np.unique([trans.transition.name for trans in sorted_transitions])
transition_names.sort()

calibration_transition = R0_F1_1_2_F_1
calibration = get_offset(sorted_transitions, calibration_transition)

# Constants
IR_UV_CONVERSION_FACTOR = 4
DEFAULT_ENERGY_RANGE = 300


# Initialize session state
def initialize_session_state():
    """Initialize session state variables with default values."""
    defaults = {
        "energy_min_val": -DEFAULT_ENERGY_RANGE,
        "energy_max_val": DEFAULT_ENERGY_RANGE,
        "prev_ir_uv": "IR",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session_state()


def adjust_energy_range_for_mode(new_mode: str) -> None:
    """Adjust energy range when switching between IR and UV modes."""
    if st.session_state["prev_ir_uv"] != new_mode:
        factor = (
            IR_UV_CONVERSION_FACTOR if new_mode == "UV" else 1 / IR_UV_CONVERSION_FACTOR
        )
        st.session_state["energy_min_val"] = int(
            st.session_state["energy_min_val"] * factor
        )
        st.session_state["energy_max_val"] = int(
            st.session_state["energy_max_val"] * factor
        )
        st.session_state["prev_ir_uv"] = new_mode


with st.sidebar:
    st.title("Transition finder")
    transition_selector = st.selectbox(label="Transition", options=transition_names)
    transition_types = st.multiselect(
        label="Transition Types",
        options=["R", "P", "Q", "S", "O"],
        default=["R", "P", "Q", "S", "O"],
    )

    ir_uv = st.selectbox(label="UV or IR", options=["IR", "UV"], index=0)
    adjust_energy_range_for_mode(ir_uv)

    col1, col2 = st.columns(2)
    with col1:
        st.number_input(label="MHz", max_value=0, step=1, key="energy_min_val")
    with col2:
        st.number_input(label="MHz", min_value=0, step=1, key="energy_max_val")

    energy_min = st.session_state["energy_min_val"]
    energy_max = st.session_state["energy_max_val"]
    cesium_frequency = st.number_input(
        label="Cesium Frequency [GHz]",
        value=calibration_transition.cesium_frequency / 1e3,
        step=1e-3,
        format="%.3f",
    )
    calibration -= calibration_transition.cesium_frequency - cesium_frequency * 1e3
    rotational_temperature = st.number_input(
        label="Rotational Temperature [K]",
        value=6.5,
        step=0.1,
        format="%.1f",
    )


def generate_plot_dataframe():
    thermal_population: npt.NDArray[np.floating] = utils.population.thermal_population(
        np.arange(11), T=rotational_temperature
    )
    error = False
    if len(transition_selector) == 0:
        st.error("No transitions selected")
        error = True
    if energy_min == energy_max:
        st.error("Set valid energy span")
        error = True
    if transition_types and transition_selector[0] not in transition_types:
        st.error(
            f"Selected transition type '{transition_selector[0]}' is not in the filter. Please select '{transition_selector[0]}' in Transition Types."
        )
        error = True
    if error:
        return

    transition_parsed = parse_transition(transition_selector)

    fig = generate_plot(
        transition_parsed,
        sorted_transitions,
        thermal_population,
        (energy_min, energy_max),
        ir_uv,
        transition_types,
    )

    st.plotly_chart(fig, width="stretch")

    df = generate_dataframe_transitions(
        transition_parsed,
        sorted_transitions,
        (energy_min, energy_max),
        ir_uv,
        calibration=calibration,
    )

    # Filter by transition types
    if transition_types:
        df = df[df.index.str[0].isin(transition_types)]

    style = df.style.format(
        formatter={
            "Δ frequency [IR, MHz]": "{:.1f}",
            "frequency [IR, GHz]": "{:.3f}",
            "Δ frequency [UV, MHz]": "{:.1f}",
            "frequency [UV, GHz]": "{:.3f}",
            "photons": "{:.2f}",
        }
    )

    st.table(style)
    df = generate_dataframe_branching(
        transition_parsed,
        sorted_transitions,
        (energy_min, energy_max),
        ir_uv,
    )

    # Filter by transition types
    if transition_types:
        df = df[df.index.str[0].isin(transition_types)]

    # Format all columns
    format_dict = {}
    for col in df.columns:
        if col.startswith("J = "):
            format_dict[col] = "{:.4f}"

    style = df.style.format(formatter=format_dict)

    st.title("Branching Ratios")
    st.table(style)


generate_plot_dataframe()
