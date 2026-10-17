

from datetime import datetime

import matplotlib.pyplot as plt

from domain.model import FaultFrequencyModel, FaultFrequencyType, MachineModel, MachineSize, MachineType, Measurement, MeasurementType, Spectrum, SpectrumPeak
from domain.data_generator import DataGenerator
from story_teller.story_teller import tell_story

bearing_fault_frequency = FaultFrequencyModel(
    name="Bearing Fault",
    type=FaultFrequencyType.Bearing,
    fault_frequency_in_orders=4.0
)

gear_fault_frequency = FaultFrequencyModel(
    name="Gear Fault",
    type=FaultFrequencyType.Gear,
    fault_frequency_in_orders=15.0
)


peak_near_bearing_fault = SpectrumPeak(
    order=bearing_fault_frequency.fault_frequency_in_orders * 1.151,
    amplitude=0.1
)

peak_near_gear_fault = SpectrumPeak(
    order=gear_fault_frequency.fault_frequency_in_orders * 1.06,
    amplitude=0.1
)

data_generator = DataGenerator(
    base_signal_f=50.0,
    sampling_frequency=1000.0,
    rms=0.2,
    sample_count=2**13,
    fmax=5000.0,
    rpm=1500.0,
    peaks=[peak_near_bearing_fault, peak_near_gear_fault]
)
base_signal_peak = SpectrumPeak(
    order=data_generator.base_signal_f / (data_generator.rpm / 60),
    amplitude=2**0.5 * data_generator.rms,
)
waveform = data_generator.generate_data()
data_generator.nosify(1)  # Add 5% Gaussian noise to the waveform
frequency, amplitude = data_generator.get_spectrum()

if waveform:
    time = [i / data_generator.sampling_frequency for i in range(len(waveform))]
    figure, (waveform_axes, spectrum_axes) = plt.subplots(2, 1, figsize=(10, 7))

    waveform_axes.plot(time, waveform)
    waveform_axes.set_xlabel("Time (s)")
    waveform_axes.set_ylabel("Amplitude")
    waveform_axes.set_title("Waveform")
    waveform_axes.grid(True)

    spectrum_axes.plot(frequency, amplitude)
    for fault_frequency in (bearing_fault_frequency, gear_fault_frequency):
        spectrum_axes.axvline(
            fault_frequency.fault_frequency_in_orders,
            color="red",
            linestyle="--",
            label=fault_frequency.name,
        )
        spectrum_axes.text(
            fault_frequency.fault_frequency_in_orders,
            1.0,
            f" {fault_frequency.name}",
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

    figure.tight_layout()
    plt.show()



machine = MachineModel(
    name="Train #1",
    type=MachineType.MotorAc,
    size=MachineSize.Medium,
    fault_frequencies=[bearing_fault_frequency, gear_fault_frequency]
)

total_energy = data_generator.get_total_energy()
crest_factor = data_generator.get_crest_factor()
last_overall_rms = data_generator.get_last_overall_rms(std_dev=0.05)

spectrum = Spectrum(
    peaks=[base_signal_peak, peak_near_bearing_fault, peak_near_gear_fault],
    total_energy=total_energy,
    last_overall_rms=last_overall_rms,
    waveform_crest_factor=crest_factor,
    timestamp=datetime.now(),
    fmax_order=data_generator.get_fmax_order()
)

measurement = Measurement(
    type=MeasurementType.Velocity,
    value=spectrum,
    machine=machine,
)

story = tell_story(measurement)


print(story)