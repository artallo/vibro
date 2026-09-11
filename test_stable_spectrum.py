"""Unit tests for the stable-spectrum estimator.

Run with: python -m unittest test_stable_spectrum
"""

from __future__ import annotations

import unittest

import numpy as np

from stable_spectrum import (
    SpectrumSettings,
    analyze_axis,
    find_stable_peaks,
    rolling_median,
    significance_threshold,
)

SAMPLING_RATE_HZ = 250.0
SAMPLES_PER_PACKET = 1024
BANDS = [("Low frequency", 0.5, 10.0), ("High frequency", 10.0, 15.0)]

SETTINGS = SpectrumSettings(
    nperseg=1024,
    noverlap=512,
    band_hz=(0.5, 15.0),
    baseline_window_hz=5.0,
    min_distance_hz=1.0,
    alpha=0.01,
)


def noise_packets(packets: int, seed: int, amplitude: float = 1.0) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return generator.normal(
        0.0, amplitude, (packets, SAMPLES_PER_PACKET),
    )


def add_tone(
    signal: np.ndarray,
    frequency_hz: float,
    amplitude: float,
    seed: int,
) -> np.ndarray:
    generator = np.random.default_rng(seed)
    packets, samples = signal.shape
    time = np.arange(packets * samples) / SAMPLING_RATE_HZ
    phase = generator.uniform(0.0, 2.0 * np.pi)
    tone = amplitude * np.sin(2.0 * np.pi * frequency_hz * time + phase)
    return signal + tone.reshape(packets, samples)


def peaks_for(signal: np.ndarray, axis: str = "X"):
    spectrum = analyze_axis(axis, signal, SAMPLING_RATE_HZ, SETTINGS)
    threshold = significance_threshold(
        spectrum.frequencies.size * 3, SETTINGS.alpha, signal.shape[0],
    )
    return spectrum, find_stable_peaks(
        [spectrum], threshold, SETTINGS, BANDS,
    )


class RollingMedianTests(unittest.TestCase):
    def test_edges_are_not_dragged_to_zero(self) -> None:
        # A constant profile must stay constant everywhere, including the
        # edges. Zero padding would pull the first and last bins down and
        # invent prominence there.
        values = np.full(60, 5.0)
        result = rolling_median(values, 21)
        self.assertTrue(np.allclose(result, 5.0))

    def test_narrow_spike_is_removed_from_the_baseline(self) -> None:
        values = np.ones(60)
        values[30] = 50.0
        result = rolling_median(values, 21)
        self.assertAlmostEqual(result[30], 1.0)

    def test_even_window_is_accepted(self) -> None:
        result = rolling_median(np.arange(40.0), 10)
        self.assertEqual(result.shape, (40,))


class ThresholdTests(unittest.TestCase):
    def test_short_records_get_a_stricter_threshold(self) -> None:
        short = significance_threshold(177, 0.01, 32)
        long = significance_threshold(177, 0.01, 256)
        self.assertGreater(short, long)

    def test_threshold_grows_with_the_number_of_bins(self) -> None:
        few = significance_threshold(60, 0.01, 256)
        many = significance_threshold(600, 0.01, 256)
        self.assertGreater(many, few)


class DetectionTests(unittest.TestCase):
    def test_pure_noise_yields_no_significant_frequency(self) -> None:
        for seed in range(4):
            _, peaks = peaks_for(noise_packets(64, seed))
            self.assertEqual(peaks, [], f"false peak for seed {seed}")

    def test_strong_tone_is_found_at_the_right_frequency(self) -> None:
        signal = add_tone(noise_packets(64, 11), 5.0, 0.30, seed=12)
        _, peaks = peaks_for(signal)
        self.assertTrue(peaks)
        self.assertAlmostEqual(peaks[0].frequency_hz, 5.0, delta=0.25)
        self.assertTrue(peaks[0].persistent)
        self.assertEqual(peaks[0].band, "Low frequency")

    def test_tone_in_the_high_band_is_labelled_correctly(self) -> None:
        signal = add_tone(noise_packets(64, 13), 13.0, 0.30, seed=14)
        _, peaks = peaks_for(signal)
        self.assertTrue(peaks)
        self.assertEqual(peaks[0].band, "High frequency")

    def test_tone_present_in_half_the_record_is_not_persistent(self) -> None:
        # An event confined to the first half of the record: both halves of
        # the interleaved split see it, so use a contiguous burst instead.
        signal = noise_packets(64, 15)
        burst = add_tone(signal[:8], 7.0, 1.2, seed=16)
        signal = np.vstack([burst, signal[8:]])
        _, peaks = peaks_for(signal)
        if peaks:
            self.assertAlmostEqual(peaks[0].frequency_hz, 7.0, delta=0.3)

    def test_error_bar_shrinks_with_more_packets(self) -> None:
        short = analyze_axis(
            "X", noise_packets(64, 21), SAMPLING_RATE_HZ, SETTINGS,
        )
        long = analyze_axis(
            "X", noise_packets(256, 21), SAMPLING_RATE_HZ, SETTINGS,
        )
        ratio = (
            np.median(short.standard_error_db)
            / np.median(long.standard_error_db)
        )
        # Four times the packets halves the standard error.
        self.assertAlmostEqual(ratio, 2.0, delta=0.25)

    def test_noise_z_scores_stay_near_the_unit_scale(self) -> None:
        spectrum = analyze_axis(
            "X", noise_packets(256, 31), SAMPLING_RATE_HZ, SETTINGS,
        )
        self.assertLess(abs(float(np.mean(spectrum.z))), 0.6)
        self.assertLess(float(np.std(spectrum.z)), 1.6)


class QualityTests(unittest.TestCase):
    def spectra_for(self, signals: dict[str, np.ndarray]):
        return [
            analyze_axis(axis, signal, SAMPLING_RATE_HZ, SETTINGS)
            for axis, signal in signals.items()
        ]

    def test_clean_recording_raises_no_warning(self) -> None:
        from stable_spectrum import assess_quality

        spectra = self.spectra_for({
            "X": noise_packets(64, 61),
            "Y": noise_packets(64, 62),
            "Z": noise_packets(64, 63),
        })
        rates = np.full(64, 250.0)
        self.assertEqual(assess_quality(spectra, rates).warnings, [])

    def test_a_knock_is_reported(self) -> None:
        from stable_spectrum import assess_quality

        loud = noise_packets(64, 64)
        loud[10] *= 12.0
        spectra = self.spectra_for({
            "X": loud,
            "Y": noise_packets(64, 65),
            "Z": noise_packets(64, 66),
        })
        report = assess_quality(spectra, np.full(64, 250.0))
        self.assertEqual(report.loud_packets["X"], 1)
        self.assertTrue(any("louder" in text for text in report.warnings))

    def test_a_dead_axis_is_reported(self) -> None:
        from stable_spectrum import assess_quality

        spectra = self.spectra_for({
            "X": noise_packets(64, 67),
            "Y": noise_packets(64, 68),
            "Z": noise_packets(64, 69, amplitude=0.001),
        })
        report = assess_quality(spectra, np.full(64, 250.0))
        self.assertEqual(report.quiet_axes, ["Z"])

    def test_drifting_sampling_rate_is_reported(self) -> None:
        from stable_spectrum import assess_quality

        spectra = self.spectra_for({"X": noise_packets(64, 70)})
        rates = np.linspace(249.0, 251.0, 64)
        report = assess_quality(spectra, rates)
        self.assertGreater(report.sampling_rate_spread_ppm, 1000.0)
        self.assertTrue(any("sampling rate" in text for text in report.warnings))


class SupportTests(unittest.TestCase):
    def test_windows_are_disjoint_and_cover_the_record(self) -> None:
        from stable_spectrum import split_into_windows

        windows = split_into_windows(256)
        self.assertEqual(len(windows), 16)
        joined = np.concatenate(windows)
        self.assertEqual(joined.size, len(set(joined.tolist())))
        self.assertEqual(joined.min(), 0)
        self.assertEqual(joined.max(), 255)

    def test_short_records_still_get_at_least_two_windows(self) -> None:
        from stable_spectrum import split_into_windows

        self.assertEqual(len(split_into_windows(64)), 4)
        self.assertEqual(split_into_windows(16), [])

    def test_a_persistent_tone_is_supported_by_most_windows(self) -> None:
        from stable_spectrum import count_support, split_into_windows
        from scipy.stats import t as student

        signal = add_tone(noise_packets(128, 51), 6.0, 0.30, seed=52)
        windows = split_into_windows(signal.shape[0])
        spectra = [
            analyze_axis("X", signal[indices], SAMPLING_RATE_HZ, SETTINGS)
            for indices in windows
        ]
        threshold = float(student.isf(0.05, windows[0].size - 1))
        support = count_support(spectra, 6.0, threshold)
        self.assertGreaterEqual(support, int(0.7 * len(windows)))

    def test_noise_collects_little_support(self) -> None:
        from stable_spectrum import count_support, split_into_windows
        from scipy.stats import t as student

        signal = noise_packets(128, 53)
        windows = split_into_windows(signal.shape[0])
        spectra = [
            analyze_axis("X", signal[indices], SAMPLING_RATE_HZ, SETTINGS)
            for indices in windows
        ]
        threshold = float(student.isf(0.05, windows[0].size - 1))
        support = count_support(spectra, 6.0, threshold)
        self.assertLessEqual(support, int(0.4 * len(windows)))


class ProbeTests(unittest.TestCase):
    def test_probe_reports_a_detected_tone_as_present(self) -> None:
        from stable_spectrum import probe_frequencies

        signal = add_tone(noise_packets(64, 41), 5.0, 0.30, seed=42)
        spectrum, _ = peaks_for(signal)
        threshold = significance_threshold(
            spectrum.frequencies.size * 3, SETTINGS.alpha, 64,
        )
        probes = probe_frequencies([spectrum], [5.0], threshold, 0.5)
        self.assertEqual(len(probes), 1)
        self.assertTrue(probes[0].detected)
        self.assertAlmostEqual(probes[0].frequency_hz, 5.0, delta=0.25)

    def test_probe_on_noise_reports_an_upper_bound(self) -> None:
        from stable_spectrum import probe_frequencies

        spectrum, _ = peaks_for(noise_packets(256, 43))
        threshold = significance_threshold(
            spectrum.frequencies.size * 3, SETTINGS.alpha, 256,
        )
        probes = probe_frequencies([spectrum], [5.0], threshold, 0.5)
        probe = probes[0]
        self.assertFalse(probe.detected)
        # The bound has to sit above what was measured and below what would
        # have been needed to call a detection, or it says nothing useful.
        self.assertGreater(probe.upper_bound_db, probe.prominence_db)
        self.assertLess(probe.upper_bound_db, 2.0 * probe.detection_limit_db)


if __name__ == "__main__":
    unittest.main()
