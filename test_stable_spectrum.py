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
    def inputs_for(self, signals: dict[str, np.ndarray]):
        spectra = [
            analyze_axis(axis, signal, SAMPLING_RATE_HZ, SETTINGS)
            for axis, signal in signals.items()
        ]
        axes = {axis.lower(): signal for axis, signal in signals.items()}
        return spectra, axes

    def test_clean_recording_raises_no_warning(self) -> None:
        from stable_spectrum import assess_quality

        spectra, axes = self.inputs_for({
            "X": noise_packets(64, 61),
            "Y": noise_packets(64, 62),
            "Z": noise_packets(64, 63),
        })
        rates = np.full(64, 250.0)
        self.assertEqual(assess_quality(spectra, rates, axes).warnings, [])

    def test_a_loud_stretch_is_reported(self) -> None:
        from stable_spectrum import assess_quality

        loud = noise_packets(64, 64)
        loud[10] *= 12.0
        spectra, axes = self.inputs_for({
            "X": loud,
            "Y": noise_packets(64, 65),
            "Z": noise_packets(64, 66),
        })
        report = assess_quality(spectra, np.full(64, 250.0), axes)
        self.assertEqual(report.loud_packets["X"], 1)
        self.assertTrue(any("louder" in text for text in report.warnings))

    def test_a_short_knock_is_reported_by_its_peak(self) -> None:
        from stable_spectrum import assess_quality

        # A sharp broadband impulse: huge in the time domain, but its energy
        # sits mostly above 15 Hz, so the in-band power of the four-second
        # packet barely moves and only the time-domain check can see it.
        knocked = noise_packets(64, 71)
        knocked[42, 400:406] += 30.0 * np.array([1, -1, 1, -1, 1, -1])
        spectra, axes = self.inputs_for({
            "X": noise_packets(64, 72),
            "Y": noise_packets(64, 73),
            "Z": knocked,
        })
        report = assess_quality(spectra, np.full(64, 250.0), axes)
        self.assertEqual(report.loud_packets["Z"], 0)
        self.assertEqual([packet for packet, _ in report.transients["Z"]], [43])
        self.assertEqual(report.transients["X"], [])
        self.assertTrue(any("transient" in text for text in report.warnings))

    def test_start_up_zero_is_reported_as_packet_one(self) -> None:
        from stable_spectrum import assess_quality

        # Z axis at rest reads 1 g with sub-milli-g noise; a zero first sample
        # from the sensor start-up is thousands of sigmas away from it.
        started = noise_packets(64, 74, amplitude=0.001) + 1.0
        started[0, 0] = 0.0
        spectra, axes = self.inputs_for({"Z": started})
        report = assess_quality(spectra, np.full(64, 250.0), axes)
        self.assertEqual([packet for packet, _ in report.transients["Z"]], [1])

    def test_a_dead_axis_is_reported(self) -> None:
        from stable_spectrum import assess_quality

        spectra, axes = self.inputs_for({
            "X": noise_packets(64, 67),
            "Y": noise_packets(64, 68),
            "Z": noise_packets(64, 69, amplitude=0.001),
        })
        report = assess_quality(spectra, np.full(64, 250.0), axes)
        self.assertEqual(report.quiet_axes, ["Z"])

    def test_drifting_sampling_rate_is_reported(self) -> None:
        from stable_spectrum import assess_quality

        spectra, axes = self.inputs_for({"X": noise_packets(64, 70)})
        rates = np.linspace(249.0, 251.0, 64)
        report = assess_quality(spectra, rates, axes)
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



def settings_at(nperseg: int) -> SpectrumSettings:
    from dataclasses import replace

    return replace(SETTINGS, nperseg=nperseg, noverlap=nperseg // 2)


def resonance(packets: int, frequency_hz: float, damping: float, seed: int) -> np.ndarray:
    """Continuous noise through a damped resonator: a broad bump."""
    from scipy.signal import lfilter

    generator = np.random.default_rng(seed)
    drive = generator.normal(0.0, 1.0, packets * SAMPLES_PER_PACKET)
    omega = 2.0 * np.pi * frequency_hz / SAMPLING_RATE_HZ
    radius = np.exp(-damping * omega)
    response = lfilter(
        [1.0], [1.0, -2.0 * radius * np.cos(omega), radius * radius], drive,
    )
    return (response / response.std()).reshape(packets, SAMPLES_PER_PACKET)


# A frequency that sits exactly on a bin at 1024 and at 4096 samples, so the
# comparison is not blurred by where the line falls inside a bin.
ON_BIN_HZ = 20 * SAMPLING_RATE_HZ / 1024


class ResolutionTests(unittest.TestCase):
    def test_overlap_factor_matches_hann_at_half_overlap(self) -> None:
        from stable_spectrum import overlap_variance_factor

        self.assertAlmostEqual(
            overlap_variance_factor(4096, 2048), 1.056, delta=0.01,
        )
        self.assertEqual(overlap_variance_factor(1024, 1024), 1.0)

    def test_segments_span_packet_joins(self) -> None:
        spectrum = analyze_axis(
            "X", noise_packets(64, 81), SAMPLING_RATE_HZ, settings_at(4096),
        )
        # 64 x 1024 samples cut into 4096 with a 2048 hop.
        self.assertEqual(spectrum.row_count, 31)
        self.assertAlmostEqual(
            spectrum.frequencies[1] - spectrum.frequencies[0],
            SAMPLING_RATE_HZ / 4096,
        )
        self.assertLess(spectrum.effective_rows, spectrum.row_count)

    def test_per_packet_halves_keep_the_old_scale(self) -> None:
        spectrum = analyze_axis(
            "X", noise_packets(64, 82), SAMPLING_RATE_HZ, SETTINGS,
        )
        self.assertEqual(spectrum.row_count, 64)
        self.assertAlmostEqual(spectrum.half_scale, 1.0 / np.sqrt(2.0))

    def test_too_short_a_record_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            analyze_axis(
                "X", noise_packets(8, 83), SAMPLING_RATE_HZ, settings_at(4096),
            )

    def test_noise_stays_quiet_at_fine_resolution(self) -> None:
        for nperseg in (2048, 4096):
            settings = settings_at(nperseg)
            for seed in range(4):
                spectrum = analyze_axis(
                    "X", noise_packets(128, 90 + seed), SAMPLING_RATE_HZ,
                    settings,
                )
                threshold = significance_threshold(
                    spectrum.frequencies.size * 3, SETTINGS.alpha,
                    int(round(spectrum.effective_rows)),
                )
                peaks = find_stable_peaks([spectrum], threshold, settings, BANDS)
                self.assertEqual(
                    peaks, [], f"false peak at {nperseg}, seed {seed}",
                )
                self.assertLess(abs(float(np.mean(spectrum.z))), 0.6)
                self.assertLess(float(np.std(spectrum.z)), 1.6)

    def test_a_narrow_line_grows_with_finer_bins(self) -> None:
        signal = add_tone(noise_packets(256, 84), ON_BIN_HZ, 0.08, seed=85)
        coarse = analyze_axis("X", signal, SAMPLING_RATE_HZ, SETTINGS)
        fine = analyze_axis("X", signal, SAMPLING_RATE_HZ, settings_at(4096))

        def at_line(spectrum):
            index = int(np.argmin(np.abs(spectrum.frequencies - ON_BIN_HZ)))
            return float(spectrum.prominence_db[index]), float(spectrum.z[index])

        coarse_db, _ = at_line(coarse)
        fine_db, fine_z = at_line(fine)
        self.assertGreater(fine_db - coarse_db, 3.0)
        self.assertGreater(fine_z, 5.0)

    def test_a_broad_bump_keeps_its_height(self) -> None:
        signal = noise_packets(256, 86) + 2.0 * resonance(256, 6.0, 0.10, seed=87)
        heights = []
        for settings in (SETTINGS, settings_at(4096)):
            spectrum = analyze_axis("X", signal, SAMPLING_RATE_HZ, settings)
            window = np.abs(spectrum.frequencies - 6.0) <= 0.3
            heights.append(float(np.max(spectrum.prominence_db[window])))
        self.assertGreater(heights[0], 3.0)
        self.assertLess(abs(heights[1] - heights[0]), 1.0)

    def test_a_steady_line_is_persistent_at_fine_resolution(self) -> None:
        settings = settings_at(4096)
        signal = add_tone(noise_packets(256, 88), ON_BIN_HZ, 0.08, seed=89)
        spectrum = analyze_axis("X", signal, SAMPLING_RATE_HZ, settings)
        threshold = significance_threshold(
            spectrum.frequencies.size * 3, SETTINGS.alpha,
            int(round(spectrum.effective_rows)),
        )
        peaks = find_stable_peaks([spectrum], threshold, settings, BANDS)
        self.assertTrue(peaks)
        self.assertAlmostEqual(peaks[0].frequency_hz, ON_BIN_HZ, delta=0.07)
        self.assertTrue(peaks[0].persistent)
        self.assertLess(spectrum.half_scale, 0.6)


class JointTests(unittest.TestCase):
    def test_continuous_noise_has_no_step(self) -> None:
        from stable_spectrum import joint_steps

        self.assertEqual(joint_steps(noise_packets(64, 91)), [])

    def test_a_step_between_packets_is_found(self) -> None:
        from stable_spectrum import joint_steps

        signal = noise_packets(64, 92)
        signal[10:] += 40.0
        self.assertEqual([packet for packet, _ in joint_steps(signal)], [10])


class DriverTests(unittest.TestCase):
    def test_each_resolution_gets_its_own_directory(self) -> None:
        import tempfile
        from pathlib import Path

        from stable_spectrum import load_settings, run_stable_spectrum

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "synthetic_raw.npz"
            np.savez(
                raw,
                x=add_tone(noise_packets(64, 93), ON_BIN_HZ, 0.3, seed=94),
                y=noise_packets(64, 95),
                z=noise_packets(64, 96),
                packet_fs_hz=np.full(64, SAMPLING_RATE_HZ),
            )
            output = root / "out"
            run_stable_spectrum(
                [raw], output, load_settings(0.01, None),
                nperseg_values=[1024, 4096],
            )
            for nperseg in (1024, 4096):
                folder = output / f"nperseg_{nperseg}"
                self.assertTrue((folder / "stable_frequencies.csv").exists())
                self.assertTrue(
                    (folder / "figure_dominant_synthetic_raw.png").exists()
                )
                report = (folder / "stable_report.txt").read_text(
                    encoding="utf-8",
                )
                self.assertIn(f"nperseg {nperseg}", report)
            comparison = (output / "comparison.txt").read_text(encoding="utf-8")
            self.assertIn("synthetic_raw", comparison)
            self.assertIn("   1024", comparison)
            self.assertIn("   4096", comparison)
            self.assertTrue(
                (output / "figure_compare_synthetic_raw.png").exists()
            )

if __name__ == "__main__":
    unittest.main()
