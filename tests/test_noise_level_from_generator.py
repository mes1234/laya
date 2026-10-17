import random
from datetime import datetime, timezone

import pytest

from domain.data_generator import DataGenerator
from domain.model import (
    MachineModel,
    MachineSize,
    MachineType,
    Measurement,
    MeasurementType,
    Spectrum,
    SpectrumPeak,
)
from story_teller.story_teller import prepare_story_teller_input, tell_story

# 60 RPM makes order == Hz; all tones fall exactly on FFT bins (1024 Hz / 8192 samples).
BASE_FREQUENCY = 50.0
SIGNAL_PEAKS = [SpectrumPeak(order=100.0, amplitude=1.0), SpectrumPeak(order=200.0, amplitude=0.5)]


def build_measurement(noise_percent: float) -> Measurement:
    generator = DataGenerator(
        base_signal_f=BASE_FREQUENCY,
        sampling_frequency=1024.0,
        rms=1.0,
        sample_count=8192,
        fmax=500.0,
        rpm=60.0,
        peaks=SIGNAL_PEAKS,
    )
    generator.generate_data()
    generator.nosify(noise_percent)

    base_peak = SpectrumPeak(order=BASE_FREQUENCY, amplitude=2**0.5)
    return Measurement(
        type=MeasurementType.Velocity,
        value=Spectrum(
            timestamp=datetime(2026, 10, 17, tzinfo=timezone.utc),
            fmax_order=generator.get_fmax_order(),
            peaks=[base_peak, *SIGNAL_PEAKS],
            total_energy=generator.get_total_energy(),
            last_overall_rms=generator.get_last_overall_rms(std_dev=0.05),
            waveform_crest_factor=generator.get_crest_factor(),
        ),
        machine=MachineModel(
            name="Test machine",
            type=MachineType.MotorAc,
            size=MachineSize.Medium,
            fault_frequencies=[],
        ),
    )


@pytest.mark.parametrize(
    ("noise_percent", "expected_category", "expected_label"),
    [
        (0, 1, "Low"),
        (5, 1, "Low"),
        (8, 1, "Low"),
        (12, 2, "Mid"),
        (20, 2, "Mid"),
        (25, 2, "Mid"),
        (35, 3, "High"),
        (50, 3, "High"),
        (100, 3, "High"),
    ],
)
def test_noise_level_follows_noisify_percentage(noise_percent, expected_category, expected_label):
    measurement = build_measurement(noise_percent)

    assert prepare_story_teller_input(measurement).noise_level == expected_category
    assert f"Noise level is {expected_label}\n" in tell_story(measurement)
