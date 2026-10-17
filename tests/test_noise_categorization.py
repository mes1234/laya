from datetime import datetime, timezone

import pytest

from domain.model import (
    FaultFrequencyModel,
    FaultFrequencyType,
    MachineModel,
    MachineSize,
    MachineType,
    Measurement,
    MeasurementType,
    Spectrum,
    SpectrumPeak,
)
from story_teller.story_teller import (
    categorize_peak_correlation,
    prepare_story_teller_input,
    tell_story,
)


def make_noise_measurement(
    peak_amplitude: float,
    overall_rms: list[float] | None = None,
    waveform_crest_factor: float = 1.0,
    peaks: list[SpectrumPeak] | None = None,
    fault_frequencies: list[FaultFrequencyModel] | None = None,
) -> Measurement:
    """Build a measurement with predictable noise and trend metrics."""
    return Measurement(
        type=MeasurementType.Velocity,
        value=Spectrum(
            timestamp=datetime(2026, 10, 17, tzinfo=timezone.utc),
            fmax_order=10.0,
            peaks=peaks if peaks is not None else [SpectrumPeak(order=1.0, amplitude=peak_amplitude)],
            total_energy=100.0,
            last_overall_rms=overall_rms if overall_rms is not None else [1.0],
            waveform_crest_factor=waveform_crest_factor,
        ),
        machine=MachineModel(
            name="Test machine",
            type=MachineType.MotorAc,
            size=MachineSize.Medium,
            fault_frequencies=fault_frequencies if fault_frequencies is not None else [],
        ),
    )


@pytest.mark.parametrize(
    ("peak_amplitude", "expected_category", "expected_label"),
    [
        (9.97, 1, "Low"),
        (9.8, 2, "Mid"),
        (9.6, 2, "Mid"),
        (9.5, 3, "High"),
        (9.0, 3, "High"),
        (0.0, 3, "High"),
    ],
)
def test_tell_story_categorizes_noise_level(peak_amplitude, expected_category, expected_label):
    measurement = make_noise_measurement(peak_amplitude)

    input_data = prepare_story_teller_input(measurement)
    story = tell_story(measurement)

    assert type(input_data.noise_level) is int
    assert input_data.noise_level == expected_category
    assert f"Noise level is {expected_label}\n" in story


def test_tell_story_lists_only_loud_correlated_peaks_by_fault_frequency():
    fault_frequencies = [
        FaultFrequencyModel("Bearing fault", FaultFrequencyType.Bearing, 4.0),
        FaultFrequencyModel("Gear fault", FaultFrequencyType.Gear, 10.0),
        FaultFrequencyModel("Shaft fault", FaultFrequencyType.Shaft, 20.0),
    ]
    measurement = make_noise_measurement(
        peak_amplitude=8.0,
        peaks=[
            SpectrumPeak(order=4.1, amplitude=5.0),
            SpectrumPeak(order=10.9, amplitude=5.0),
            SpectrumPeak(order=15.0, amplitude=5.0),
            SpectrumPeak(order=4.0, amplitude=0.2),
        ],
        fault_frequencies=fault_frequencies,
    )

    story = tell_story(measurement)

    assert "High correlated peak near Bearing fault\n" in story
    assert "Moderate correlated peak near Gear fault\n" in story
    assert "Shaft fault" not in story
    assert story.count("correlated peak near") == 2


@pytest.mark.parametrize(
    ("metric", "metric_value", "expected_category", "expected_label"),
    [
        ("std", 0.24, 1, "Low"),
        ("std", 0.25, 2, "Mid"),
        ("std", 0.75, 2, "Mid"),
        ("std", 0.7501, 3, "High"),
        ("std", 0.9, 3, "High"),
        ("crest", 1.41, 1, "Low"),
        ("crest", 2.99, 1, "Low"),
        ("crest", 3.0, 2, "Mid"),
        ("crest", 5.0, 2, "Mid"),
        ("crest", 5.01, 3, "High"),
        ("crest", 8.0, 3, "High"),
    ],
)
def test_tell_story_categorizes_trend_metrics(metric, metric_value, expected_category, expected_label):
    if metric == "std":
        overall_rms = [1.0 - metric_value, 1.0 + metric_value]
        waveform_crest_factor = 1.0
        story_metric = "Trend standard deviation"
        category_attribute = "trend_standard_deviation"
    else:
        overall_rms = [1.0, 1.0]
        waveform_crest_factor = metric_value
        story_metric = "Crest factor"
        category_attribute = "crest_factor"

    measurement = make_noise_measurement(
        peak_amplitude=10.0,
        overall_rms=overall_rms,
        waveform_crest_factor=waveform_crest_factor,
    )

    input_data = prepare_story_teller_input(measurement)
    story = tell_story(measurement)

    assert type(getattr(input_data, category_attribute)) is int
    assert getattr(input_data, category_attribute) == expected_category
    assert f"{story_metric} is {expected_label}\n" in story


@pytest.fixture
def fault_frequency():
    return FaultFrequencyModel(
        name="Bearing fault",
        type=FaultFrequencyType.Bearing,
        fault_frequency_in_orders=100.0,
    )


@pytest.mark.parametrize(
    ("peak_order", "expected"),
    [
        (100.0, "High"),
        (105.0, "High"),
        (95.0, "High"),
        (105.5, "Moderate"),
        (90.0, "Moderate"),
        (110.0, "Moderate"),
        (110.5, "No"),
        (89.0, "No"),
    ],
)
def test_categorize_peak_correlation(fault_frequency, peak_order, expected):
    peak = SpectrumPeak(order=peak_order, amplitude=1.0)

    assert categorize_peak_correlation(peak, fault_frequency) == expected


def test_noise_level_is_noise_to_signal_amplitude_ratio():
    from story_teller.story_teller import estimate_noise_level

    assert estimate_noise_level(make_noise_measurement(8.0)) == pytest.approx((36 / 64) ** 0.5)
    assert estimate_noise_level(make_noise_measurement(10.0)) == 0.0
    assert estimate_noise_level(make_noise_measurement(20.0)) == 0.0
