import math

import pytest

from domain.data_generator import DataGenerator


def test_generate_data_returns_rms_scaled_samples_at_sampling_frequency():
    generator = DataGenerator(
        base_signal_f=1.0,
        sampling_frequency=4.0,
        rms=2.0,
        sample_count=4,
        fmax=2.0,
        rpm=60.0,
        peaks=[],
    )

    waveform = generator.generate_data()

    assert len(waveform) == 4
    assert waveform == pytest.approx(
        [0.0, 2.0 * math.sqrt(2), 0.0, -2.0 * math.sqrt(2)]
    )


def test_generate_data_requires_positive_sampling_frequency():
    generator = DataGenerator(1.0, 0.0, 1.0, 4, 2.0, 60.0, [])

    with pytest.raises(ValueError, match="sampling_frequency must be greater than zero"):
        generator.generate_data()


def test_generate_data_rejects_negative_sample_count():
    generator = DataGenerator(1.0, 4.0, 1.0, -1, 2.0, 60.0, [])

    with pytest.raises(ValueError, match="sample_count cannot be negative"):
        generator.generate_data()


def test_get_spectrum_peaks_at_base_frequency():
    from domain.data_generator import DataGenerator

    generator = DataGenerator(50.0, 1000.0, 1.0, 1000, 500.0, 1500.0, [])
    generator.generate_data()
    frequencies, amplitudes = generator.get_spectrum()

    assert len(frequencies) == len(amplitudes) == 501
    peak_index = amplitudes.index(max(amplitudes))
    assert abs(frequencies[peak_index] - 2.0) < 1e-9
    assert abs(amplitudes[peak_index] - 2 ** 0.5) < 1e-6


def test_get_spectrum_empty_for_zero_samples():
    from domain.data_generator import DataGenerator

    assert DataGenerator(50.0, 1000.0, 1.0, 0, 500.0, 1500.0, []).get_spectrum() == ([], [])


def test_peaks_are_added_to_waveform_at_order_frequency():
    from domain.data_generator import DataGenerator
    from domain.model import SpectrumPeak

    peak = SpectrumPeak(order=4.0, amplitude=3.0)
    generator = DataGenerator(50.0, 1000.0, 1.0, 1000, 500.0, 1500.0, [peak])
    generator.generate_data()
    frequencies, amplitudes = generator.get_spectrum()

    assert abs(amplitudes[min(range(len(frequencies)), key=lambda i: abs(frequencies[i] - 2.0))] - 2 ** 0.5) < 1e-6
    assert abs(amplitudes[min(range(len(frequencies)), key=lambda i: abs(frequencies[i] - 4.0))] - 3.0) < 1e-6


def test_first_order_is_at_one_for_60_rpm():
    from domain.data_generator import DataGenerator

    generator = DataGenerator(1.0, 64.0, 1.0, 64, 32.0, 60.0, [])
    generator.generate_data()
    orders, amplitudes = generator.get_spectrum()

    assert orders[amplitudes.index(max(amplitudes))] == 1.0


def test_nosify_adds_noise_scaled_by_percentage():
    import math
    import pytest
    from domain.data_generator import DataGenerator

    generator = DataGenerator(50.0, 1000.0, 1.0, 10000, 500.0, 1500.0, [])
    clean = list(generator.generate_data())
    noisy = generator.nosify(10)

    assert noisy != clean
    noise = [n - c for n, c in zip(noisy, clean)]
    noise_rms = math.sqrt(sum(x * x for x in noise) / len(noise))
    assert noise_rms == pytest.approx(0.1, rel=0.1)


def test_nosify_zero_percent_keeps_waveform_and_rejects_negative():
    import pytest
    from domain.data_generator import DataGenerator

    generator = DataGenerator(50.0, 1000.0, 1.0, 100, 500.0, 1500.0, [])
    clean = list(generator.generate_data())

    assert generator.nosify(0) == clean
    with pytest.raises(ValueError):
        generator.nosify(-1)


def test_get_total_energy_is_sum_of_squared_amplitudes():
    import pytest
    from domain.data_generator import DataGenerator

    generator = DataGenerator(50.0, 1000.0, 1.0, 1000, 500.0, 1500.0, [])
    generator.generate_data()
    _, amplitudes = generator.get_spectrum()

    assert generator.get_total_energy() == pytest.approx(sum(a * a for a in amplitudes))
    assert generator.get_total_energy() == pytest.approx(2.0)


def test_get_total_energy_is_zero_without_samples():
    from domain.data_generator import DataGenerator

    assert DataGenerator(50.0, 1000.0, 1.0, 0, 500.0, 1500.0, []).get_total_energy() == 0


def test_get_crest_factor_of_sine_is_sqrt_two():
    import pytest
    from domain.data_generator import DataGenerator

    generator = DataGenerator(50.0, 1000.0, 1.0, 1000, 500.0, 1500.0, [])
    generator.generate_data()

    assert generator.get_crest_factor() == pytest.approx(2 ** 0.5, rel=1e-3)


def test_get_crest_factor_rejects_empty_waveform():
    import pytest
    from domain.data_generator import DataGenerator

    generator = DataGenerator(50.0, 1000.0, 1.0, 0, 500.0, 1500.0, [])
    generator.generate_data()

    with pytest.raises(ValueError):
        generator.get_crest_factor()


def test_get_last_overall_rms_length_and_values():
    import pytest
    from domain.data_generator import DataGenerator

    generator = DataGenerator(50.0, 1000.0, 1.0, 1000, 500.0, 1500.0, [])
    generator.generate_data()

    for _ in range(50):
        values = generator.get_last_overall_rms(0.0)
        assert 4 <= len(values) <= 20
        assert values == [pytest.approx(1.0, rel=1e-3)] * len(values)

    assert len(set(generator.get_last_overall_rms(0.5))) > 1
    with pytest.raises(ValueError):
        generator.get_last_overall_rms(-1)


def test_get_fmax_order_divides_fmax_by_shaft_frequency():
    import pytest
    from domain.data_generator import DataGenerator

    generator = DataGenerator(50.0, 1000.0, 1.0, 1000, 500.0, 1500.0, [])

    assert generator.get_fmax_order() == pytest.approx(20.0)
    generator.rpm = 0
    with pytest.raises(ValueError):
        generator.get_fmax_order()


def test_get_last_overall_rms_has_requested_std_dev():
    import statistics
    import pytest
    from domain.data_generator import DataGenerator

    generator = DataGenerator(50.0, 1000.0, 1.0, 1000, 500.0, 1500.0, [])
    generator.generate_data()

    for _ in range(20):
        values = generator.get_last_overall_rms(0.1)
        assert statistics.pstdev(values) == pytest.approx(0.1, rel=1e-6)
        assert statistics.fmean(values) == pytest.approx(generator.get_waveform_rms(), rel=1e-6)
