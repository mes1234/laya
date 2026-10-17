import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, Slider

from domain.data_generator import DataGenerator
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
from story_teller.story_teller import tell_story

BEARING_FAULT = FaultFrequencyModel("Bearing Fault", FaultFrequencyType.Bearing, 4.0)
GEAR_FAULT = FaultFrequencyModel("Gear Fault", FaultFrequencyType.Gear, 15.0)

SAMPLING_FREQUENCY = 1000.0
SAMPLE_COUNT = 2**13
FMAX = 5000.0
RPM = 1500.0
BASE_SIGNAL_F = 50.0
DATA_FILE = Path(__file__).with_name("data.json")
CORRECT_OPTIONS = ["Normal", "Noisy", "Gear", "Bearing"]


def append_record(story: str, correct: str, path: Path = DATA_FILE) -> None:
    """Append one JSON object per line (JSON Lines) to the data file."""
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps({"story": story.strip().replace("\n", ";"), "correct": correct}) + "\n")


# Noise percentages that land in the Low / Mid / High noise-level bands.
NOISE_PERCENT_BY_SIZE = {
    MachineSize.Small: 5.0,
    MachineSize.Medium: 20.0,
    MachineSize.Large: 50.0,
}


@dataclass
class Settings:
    motor_type: MachineType = MachineType.MotorAc
    motor_size: MachineSize = MachineSize.Medium
    noise_percent: float = NOISE_PERCENT_BY_SIZE[MachineSize.Medium]
    base_rms: float = 0.2
    trend_std_percent: float = 10.0
    gear_amplitude: float = 0.1
    gear_ratio: float = 1.06
    bearing_amplitude: float = 0.1
    bearing_ratio: float = 1.15


@dataclass
class Scene:
    time: list[float]
    waveform: list[float]
    orders: list[float]
    amplitudes: list[float]
    story: str


def build_scene(settings: Settings) -> Scene:
    bearing_peak = SpectrumPeak(
        order=BEARING_FAULT.fault_frequency_in_orders * settings.bearing_ratio,
        amplitude=settings.bearing_amplitude,
    )
    gear_peak = SpectrumPeak(
        order=GEAR_FAULT.fault_frequency_in_orders * settings.gear_ratio,
        amplitude=settings.gear_amplitude,
    )

    generator = DataGenerator(
        base_signal_f=BASE_SIGNAL_F,
        sampling_frequency=SAMPLING_FREQUENCY,
        rms=settings.base_rms,
        sample_count=SAMPLE_COUNT,
        fmax=FMAX,
        rpm=RPM,
        peaks=[bearing_peak, gear_peak],
    )
    generator.generate_data()
    generator.nosify(settings.noise_percent)
    orders, amplitudes = generator.get_spectrum()

    base_peak = SpectrumPeak(
        order=generator.base_signal_f / (generator.rpm / 60),
        amplitude=2**0.5 * generator.rms,
    )
    measurement = Measurement(
        type=MeasurementType.Velocity,
        value=Spectrum(
            timestamp=datetime.now(),
            fmax_order=generator.get_fmax_order(),
            peaks=[base_peak, bearing_peak, gear_peak],
            total_energy=generator.get_total_energy(),
            last_overall_rms=generator.get_last_overall_rms(
                std_dev=generator.get_waveform_rms() * settings.trend_std_percent / 100
            ),
            waveform_crest_factor=generator.get_crest_factor(),
        ),
        machine=MachineModel(
            name="Interactive machine",
            type=settings.motor_type,
            size=settings.motor_size,
            fault_frequencies=[BEARING_FAULT, GEAR_FAULT],
        ),
    )

    time = [i / SAMPLING_FREQUENCY for i in range(len(generator.wf))]
    return Scene(time, generator.wf, orders, amplitudes, tell_story(measurement))


def main() -> None:
    settings = Settings()
    figure = plt.figure(figsize=(14, 8))
    figure.canvas.manager.set_window_title("Laya interactive playground")

    waveform_axes = figure.add_axes([0.40, 0.76, 0.57, 0.18])
    spectrum_axes = figure.add_axes([0.40, 0.40, 0.57, 0.26])
    story_axes = figure.add_axes([0.40, 0.04, 0.57, 0.24])
    story_axes.axis("off")
    story_axes.set_title("Story teller", loc="left")
    story_text = story_axes.text(
        0.0, 1.0, "", va="top", ha="left", family="monospace", transform=story_axes.transAxes
    )

    type_axes = figure.add_axes([0.03, 0.80, 0.2, 0.13])
    type_axes.set_title("Motor type", loc="left")
    type_radio = RadioButtons(
        type_axes,
        [member.value for member in MachineType],
        active=list(MachineType).index(settings.motor_type),
    )

    size_axes = figure.add_axes([0.03, 0.60, 0.2, 0.13])
    size_axes.set_title("Motor size", loc="left")
    size_radio = RadioButtons(
        size_axes,
        [member.value for member in MachineSize],
        active=list(MachineSize).index(settings.motor_size),
    )

    correct_axes = figure.add_axes([0.03, 0.01, 0.2, 0.11])
    correct_axes.set_title("Correct", loc="left")
    correct_radio = RadioButtons(correct_axes, CORRECT_OPTIONS)

    save_axes = figure.add_axes([0.27, 0.05, 0.1, 0.05])
    save_button = Button(save_axes, "Save")
    save_status = figure.text(0.27, 0.025, "", fontsize=8)
    current_story = {"text": ""}

    def on_save(_event) -> None:
        append_record(current_story["text"], correct_radio.value_selected)
        save_status.set_text(f"Saved: {correct_radio.value_selected}")
        figure.canvas.draw_idle()

    save_button.on_clicked(on_save)

    def add_slider(y: float, label: str, low: float, high: float, value: float) -> Slider:
        axes = figure.add_axes([0.13, y, 0.18, 0.03])
        return Slider(axes, label, low, high, valinit=value)

    sliders = {
        "noise_percent": add_slider(0.52, "Noise %", 0.0, 100.0, settings.noise_percent),
        "base_rms": add_slider(0.47, "Base RMS", 0.05, 2.0, settings.base_rms),
        "trend_std_percent": add_slider(0.42, "Trend std %", 0.0, 100.0, settings.trend_std_percent),
        "gear_amplitude": add_slider(0.34, "Gear amp", 0.0, 1.0, settings.gear_amplitude),
        "gear_ratio": add_slider(0.29, "Gear ratio", 0.8, 1.2, settings.gear_ratio),
        "bearing_amplitude": add_slider(0.23, "Bearing amp", 0.0, 1.0, settings.bearing_amplitude),
        "bearing_ratio": add_slider(0.19, "Bearing ratio", 0.8, 1.2, settings.bearing_ratio),
    }

    def redraw(_=None) -> None:
        for name, slider in sliders.items():
            setattr(settings, name, slider.val)
        settings.motor_type = MachineType(type_radio.value_selected)
        settings.motor_size = MachineSize(size_radio.value_selected)
        scene = build_scene(settings)

        waveform_axes.clear()
        waveform_axes.plot(scene.time, scene.waveform)
        waveform_axes.set_xlabel("Time (s)", labelpad=2)
        waveform_axes.set_ylabel("Amplitude")
        waveform_axes.set_title("Waveform")
        waveform_axes.grid(True)

        spectrum_axes.clear()
        spectrum_axes.plot(scene.orders, scene.amplitudes)
        for fault in (BEARING_FAULT, GEAR_FAULT):
            spectrum_axes.axvline(fault.fault_frequency_in_orders, color="red", linestyle="--")
            spectrum_axes.text(
                fault.fault_frequency_in_orders,
                1.0,
                f" {fault.name}",
                color="red",
                rotation=90,
                va="top",
                ha="left",
                transform=spectrum_axes.get_xaxis_transform(),
            )
        spectrum_axes.set_xlabel("Order")
        spectrum_axes.set_ylabel("Amplitude")
        spectrum_axes.set_title("Order spectrum")
        spectrum_axes.grid(True)

        story_text.set_text(scene.story)
        current_story["text"] = scene.story
        figure.canvas.draw_idle()

    for slider in sliders.values():
        slider.on_changed(redraw)
    type_radio.on_clicked(redraw)

    def on_size_selected(label: str) -> None:
        noise_slider = sliders["noise_percent"]
        target = NOISE_PERCENT_BY_SIZE[MachineSize(label)]
        if noise_slider.val != target:
            noise_slider.set_val(target)
        else:
            redraw()

    size_radio.on_clicked(on_size_selected)

    redraw()
    plt.show()


if __name__ == "__main__":
    main()
