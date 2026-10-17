from domain.model import FaultFrequencyModel, Measurement, SpectrumPeak
from dataclasses import dataclass
from statistics import fmean, pstdev

@dataclass
class StoryTellerInput:
    noise_level: int
    peaks_with_correlation: list[tuple[FaultFrequencyModel, str]]
    trend_standard_deviation: int
    crest_factor: int

def prepare_story_teller_input(measurement: Measurement) -> StoryTellerInput:
    noise_ratio = estimate_noise_level(measurement)
    noise_level = categorize_noise_ratio(noise_ratio)
    peaks_with_correlation = correlate_peaks_with_fault_frequencies(measurement, noise_ratio)
    standard_deviation, _ = summarize_overall_trend(measurement)
    mean_rms = fmean(measurement.value.last_overall_rms)
    return StoryTellerInput(
        noise_level=noise_level,
        peaks_with_correlation=peaks_with_correlation,
        trend_standard_deviation=categorize_noise_level(standard_deviation / mean_rms),
        crest_factor=categorize_crest_factor(measurement.value.waveform_crest_factor),
    )


def correlate_peaks_with_fault_frequencies(
    measurement: Measurement,
    noise_ratio: float,
) -> list[tuple[FaultFrequencyModel, str]]:
    peaks_with_correlation = []
    for peak in measurement.value.peaks:
        if peak.amplitude <= 2 * noise_ratio:
            continue
        if not measurement.machine.fault_frequencies:
            continue

        fault_frequency = min(
            measurement.machine.fault_frequencies,
            key=lambda candidate: relative_order_deviation(peak, candidate),
        )
        level = categorize_peak_correlation(peak, fault_frequency)
        if level != "No":
            peaks_with_correlation.append((fault_frequency, level))
    return peaks_with_correlation


def tell_story(measurement: Measurement) -> str:
    input_data = prepare_story_teller_input(measurement)
    story = f"Machine is {measurement.machine.name} of size {measurement.machine.size.value} of type {measurement.machine.type.value}\n"
    story += (
        f"Noise level is {format_category(input_data.noise_level)}\n"
    )
    for fault_frequency, level in input_data.peaks_with_correlation:
        story += f"{level} correlated peak near {fault_frequency.name}\n"
    story += (
        "Trend standard deviation is "
        f"{format_category(input_data.trend_standard_deviation)}\n"
    )
    story += (
        f"Crest factor is {format_category(input_data.crest_factor)}\n"
    )
    return story

def format_category(category: int) -> str:
    labels = {1: "Low", 2: "Mid", 3: "High"}
    return labels[category]

def estimate_noise_level(measurement: Measurement) -> float:
    """Noise-to-signal amplitude ratio: peaks are the signal, the remaining energy is noise."""
    total_energy = measurement.value.total_energy
    if total_energy == 0:
        raise ValueError("Cannot calculate noise level when total energy is zero.")

    signal_energy = sum(peak.amplitude**2 for peak in measurement.value.peaks)
    if signal_energy == 0:
        return float("inf")
    noise_energy = max(0.0, total_energy - signal_energy)
    return (noise_energy / signal_energy) ** 0.5


def categorize_crest_factor(crest_factor: float) -> int:
    """Low below 3, Mid from 3 to 5, High above 5."""
    if crest_factor < 3:
        return 1
    if crest_factor <= 5:
        return 2
    return 3


def categorize_noise_ratio(noise_ratio: float) -> int:
    """Low below 10% noise, Mid up to 30%, High above."""
    if noise_ratio < 0.10:
        return 1
    if noise_ratio <= 0.30:
        return 2
    return 3


def categorize_noise_level(noise_ratio: float) -> int:
    if noise_ratio < 0.25:
        return 1
    if noise_ratio <= 0.75:
        return 2
    return 3


def relative_order_deviation(peak: SpectrumPeak, fault_frequency: FaultFrequencyModel) -> float:
    fault_order = fault_frequency.fault_frequency_in_orders
    difference = abs(peak.order - fault_order)
    if fault_order == 0:
        return 0.0 if difference == 0 else float("inf")
    return difference / abs(fault_order)


def categorize_peak_correlation(peak: SpectrumPeak, fault_frequency: FaultFrequencyModel) -> str:
    """High within +/-5% of the fault order, Moderate within +/-10%, otherwise No."""
    deviation = relative_order_deviation(peak, fault_frequency)
    if deviation <= 0.05:
        return "High"
    if deviation <= 0.10:
        return "Moderate"
    return "No"


def summarize_overall_trend(measurement: Measurement) -> tuple[float, float]:
    overall_rms = measurement.value.last_overall_rms
    if not overall_rms:
        raise ValueError("Cannot summarize overall trend without RMS values.")

    mean_rms = fmean(overall_rms)
    if mean_rms == 0:
        raise ValueError("Cannot calculate crest factor ratio when mean RMS is zero.")

    standard_deviation = pstdev(overall_rms)
    crest_factor_ratio = measurement.value.waveform_crest_factor / mean_rms
    return standard_deviation, crest_factor_ratio
