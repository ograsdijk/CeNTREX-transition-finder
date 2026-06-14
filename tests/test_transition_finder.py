import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transition_finder import (
    _display_frequency_ghz,
    _marker_delta_from_frequency_ghz,
    _marker_frequency_ghz_from_delta,
    _number_column_config,
    _selected_line_readout_rows,
    _selected_table_rows,
)


def test_selected_table_rows_matches_displayed_dataframe_rows():
    frame = pd.DataFrame(
        [
            {"transition": "a", "frequency [IR, GHz]": 1.0},
            {"transition": "b", "frequency [IR, GHz]": 2.0},
            {"transition": "c", "frequency [IR, GHz]": 3.0},
        ]
    )

    selected = _selected_table_rows(frame, [2, 99, 0])

    assert list(selected["transition"]) == ["c", "a"]


def test_number_column_config_only_formats_existing_numeric_columns():
    column_config = _number_column_config(
        ["transition", "Δ freq [IR, MHz]", "strength"],
        {
            "Δ freq [IR, MHz]": "%.1f",
            "strength": "%.3e",
            "missing": "%.2f",
        },
    )

    assert set(column_config) == {"Δ freq [IR, MHz]", "strength"}


def test_selected_line_readout_formats_optional_fields():
    delta = chr(916)
    selected = pd.DataFrame(
        [
            {
                "transition": "R(2) F1'=7/2 F'=3",
                "frequency [UV, GHz]": 1103435.092,
                f"{delta} freq [UV, MHz]": 80.0,
                f"{delta} from 0 V/cm [UV, MHz]": 4.0,
                "strength": 1.23e-4,
                "photons": 12.3,
                "P'": "+",
                "dominant mF": "0 -> 1 (100%)",
            }
        ]
    )

    readout = _selected_line_readout_rows(selected, "UV")

    assert readout.iloc[0]["UV freq [GHz]"] == "1103435.092000"
    assert readout.iloc[0]["UV wavelength [nm]"] == "0.271690"
    assert readout.iloc[0][f"{delta} freq [MHz]"] == 80.0
    assert readout.iloc[0]["parity"] == "+"


def test_marker_frequency_delta_conversion_round_trips_ir():
    context = {
        "ir_uv": "IR",
        "reference_frequency_ir_mhz": 275858770.0,
        "calibration_offset_ir_mhz": 0.677,
    }

    delta_mhz = _marker_delta_from_frequency_ghz(275858.775677, context)
    frequency_ghz = _marker_frequency_ghz_from_delta(delta_mhz, context)

    assert delta_mhz == pytest.approx(5.0)
    assert frequency_ghz == pytest.approx(275858.775677)


def test_marker_frequency_delta_conversion_round_trips_uv():
    context = {
        "ir_uv": "UV",
        "reference_frequency_ir_mhz": 275858770.0,
        "calibration_offset_ir_mhz": 0.677,
    }

    delta_mhz = _marker_delta_from_frequency_ghz(1103435.122708, context)
    frequency_ghz = _marker_frequency_ghz_from_delta(delta_mhz, context)

    assert delta_mhz == pytest.approx(40.0)
    assert frequency_ghz == pytest.approx(1103435.122708)


def test_marker_frequency_delta_conversion_includes_reference_axis_shift_uv():
    context = {
        "ir_uv": "UV",
        "reference_frequency_ir_mhz": 1000.0,
        "calibration_offset_ir_mhz": 0.0,
        "reference_axis_shift_ir_mhz": -10.0,
    }

    displayed_frequency = _display_frequency_ghz(
        1000.0,
        ir_uv="UV",
        calibration_offset_ir_mhz=0.0,
        reference_axis_shift_ir_mhz=-10.0,
    )
    delta_mhz = _marker_delta_from_frequency_ghz(displayed_frequency, context)

    assert delta_mhz == pytest.approx(0.0)


def test_marker_frequency_delta_round_trip_with_reference_axis_shift_uv():
    context = {
        "ir_uv": "UV",
        "reference_frequency_ir_mhz": 1000.0,
        "calibration_offset_ir_mhz": 0.0,
        "reference_axis_shift_ir_mhz": -10.0,
    }

    frequency_ghz = _marker_frequency_ghz_from_delta(40.0, context)
    delta_mhz = _marker_delta_from_frequency_ghz(frequency_ghz, context)

    assert delta_mhz == pytest.approx(40.0)
