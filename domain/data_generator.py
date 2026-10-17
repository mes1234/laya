

import math
import random

import numpy as np

from domain.model import SpectrumPeak


class DataGenerator:
    def __init__(self, base_signal_f: float, sampling_frequency: float, rms: float, sample_count: int, fmax: float, rpm: float, peaks: list[SpectrumPeak]):
        self.base_signal_f = base_signal_f
        self.rms = rms
        self.sample_count = sample_count
        self.fmax = fmax
        self.sampling_frequency = sampling_frequency
        self.rpm = rpm
        self.peaks = peaks
        self.wf = []


    def generate_data(self):
        if self.sampling_frequency <= 0:
            raise ValueError("sampling_frequency must be greater than zero.")
        if self.sample_count < 0:
            raise ValueError("sample_count cannot be negative.")

        amplitude = math.sqrt(2) * self.rms
        shaft_frequency = self.rpm / 60
        self.wf = []
        for sample_index in range(self.sample_count):
            time = sample_index / self.sampling_frequency
            value = amplitude * math.sin(2 * math.pi * self.base_signal_f * time)
            for peak in self.peaks:
                peak_frequency = peak.order * shaft_frequency
                value += peak.amplitude * math.sin(2 * math.pi * peak_frequency * time)
            self.wf.append(value)

        return self.wf

    def nosify(self, prct: float):
        """Add Gaussian noise with std equal to prct percent of the waveform RMS."""
        if prct < 0:
            raise ValueError("prct cannot be negative.")
        if not self.wf:
            return self.wf

        waveform_rms = self.get_waveform_rms()
        sigma = waveform_rms * prct / 100
        self.wf = [value + random.gauss(0, sigma) for value in self.wf]
        return self.wf

    def get_spectrum(self) -> tuple[list[float], list[float]]:
        """Return one-sided spectrum as (orders, amplitudes); order 1 is the shaft speed."""
        if self.rpm <= 0:
            raise ValueError("rpm must be greater than zero.")
        waveform = self.wf
        count = len(waveform)
        if count == 0:
            return [], []

        shaft_frequency = self.rpm / 60
        frequencies = np.fft.rfftfreq(count, d=1 / self.sampling_frequency)
        amplitudes = np.abs(np.fft.rfft(waveform)) / count
        if count > 1:
            amplitudes[1:-1 if count % 2 == 0 else None] *= 2
        return (frequencies / shaft_frequency).tolist(), amplitudes.tolist()

    def get_total_energy(self) -> float:
        """Return the sum of squared spectrum amplitudes."""
        _, amplitudes = self.get_spectrum()
        return sum(amplitude * amplitude for amplitude in amplitudes)

    def get_waveform_rms(self) -> float:
        if not self.wf:
            raise ValueError("waveform is empty.")
        return math.sqrt(sum(value * value for value in self.wf) / len(self.wf))

    def get_crest_factor(self) -> float:
        """Return waveform peak absolute value divided by waveform RMS."""
        if not self.wf:
            raise ValueError("waveform is empty.")
        waveform_rms = self.get_waveform_rms()
        if waveform_rms == 0:
            raise ValueError("waveform RMS cannot be zero.")
        return max(abs(value) for value in self.wf) / waveform_rms

    def get_fmax_order(self) -> float:
        """Return fmax expressed in orders of the shaft speed."""
        if self.rpm <= 0:
            raise ValueError("rpm must be greater than zero.")
        return self.fmax / (self.rpm / 60)

    def get_last_overall_rms(self, std_dev: float) -> list[float]:
        """Return 4-20 RMS values: the waveform RMS with an exact population std_dev around it (before clamping at 0)."""
        if std_dev < 0:
            raise ValueError("std_dev cannot be negative.")
        if not self.wf:
            raise ValueError("waveform is empty.")
        waveform_rms = self.get_waveform_rms()
        length = random.randint(4, 20)
        draws = [random.gauss(0, 1) for _ in range(length)]
        mean = sum(draws) / length
        spread = math.sqrt(sum((draw - mean) ** 2 for draw in draws) / length)
        if spread == 0:
            return [waveform_rms] * length
        return [max(0.0, waveform_rms + std_dev * (draw - mean) / spread) for draw in draws]