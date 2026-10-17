from datetime import datetime, timezone

import pytest

from domain.model import (
    MachineModel,
    MachineSize,
    MachineType,
    Measurement,
    MeasurementType,
    Spectrum,
)
from story_teller.story_teller import summarize_overall_trend


def make_measurement(overall_rms: list[float], waveform_crest_factor: float) -> Measurement:
    return Measurement(
        type=MeasurementType.Velocity,
        value=Spectrum(
            timestamp=datetime(2026, 10, 17, tzinfo=timezone.utc),
            fmax_order=10.0,
            peaks=[],
            total_energy=1.0,
            last_overall_rms=overall_rms,
            waveform_crest_factor=waveform_crest_factor,
        ),
        machine=MachineModel(
            name="Pump A",
            type=MachineType.MotorAc,
            size=MachineSize.Medium,
            fault_frequencies=[],
        ),
    )


def test_summarize_overall_trend_returns_standard_deviation_and_crest_ratio():
    measurement = make_measurement([1.0, 2.0, 3.0], waveform_crest_factor=6.0)

    standard_deviation, crest_factor_ratio = summarize_overall_trend(measurement)

    assert standard_deviation == pytest.approx((2.0 / 3.0) ** 0.5)
    assert crest_factor_ratio == pytest.approx(3.0)


def test_summarize_overall_trend_requires_rms_values():
    measurement = make_measurement([], waveform_crest_factor=6.0)

    with pytest.raises(ValueError, match="without RMS values"):
        summarize_overall_trend(measurement)


def test_summarize_overall_trend_requires_nonzero_mean_rms():
    measurement = make_measurement([0.0, 0.0], waveform_crest_factor=6.0)

    with pytest.raises(ValueError, match="mean RMS is zero"):
        summarize_overall_trend(measurement)
