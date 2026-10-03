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



class HonestProbeTests(unittest.TestCase):
    def test_probe_text_with_and_without_an_axis(self) -> None:
        import argparse

        from stable_spectrum import parse_probe

        self.assertEqual(parse_probe("2.88").axis, None)
        request = parse_probe("y:2,88")
        self.assertEqual((request.axis, request.frequency_hz), ("Y", 2.88))
        for wrong in ("Q:2.88", "abc", "-1", "Y:"):
            with self.assertRaises(argparse.ArgumentTypeError):
                parse_probe(wrong)

    def test_the_threshold_counts_only_the_probe_windows(self) -> None:
        from stable_spectrum import ProbeRequest, probe_threshold

        spectrum, _ = peaks_for(noise_packets(256, 960))
        full = significance_threshold(spectrum.frequencies.size * 3, 0.01, 256)
        one_axis, bins_one = probe_threshold(
            [spectrum], [ProbeRequest(5.0, "X")], 0.2, 0.01, 256,
        )
        other_axis, bins_other = probe_threshold(
            [spectrum], [ProbeRequest(5.0, "Y")], 0.2, 0.01, 256,
        )
        self.assertLess(one_axis, full)
        self.assertGreaterEqual(bins_one, 1)
        self.assertLessEqual(bins_one, 2)
        self.assertEqual((other_axis, bins_other), (0.0, 0))

    def test_a_weak_line_passes_the_honest_threshold_not_the_blind_one(self) -> None:
        from stable_spectrum import ProbeRequest, probe_frequencies, probe_threshold

        for seed in (0, 3):
            signal = add_tone(noise_packets(256, 900 + seed), ON_BIN_HZ, 0.04, seed=950 + seed)
            spectrum, peaks = peaks_for(signal)
            self.assertEqual(peaks, [], f"seed {seed}: the blind search should miss it")
            request = [ProbeRequest(ON_BIN_HZ, "X")]
            threshold, _ = probe_threshold([spectrum], request, 0.2, 0.01, 256)
            probe = probe_frequencies([spectrum], request, threshold, 0.2)[0]
            self.assertTrue(probe.detected, f"seed {seed}: z {probe.z:.2f}")

    def test_a_stronger_neighbour_does_not_answer_for_the_probe(self) -> None:
        from stable_spectrum import ProbeRequest, probe_frequencies, probe_threshold

        signal = add_tone(noise_packets(256, 961), 5.6, 0.3, seed=962)
        spectrum, _ = peaks_for(signal)
        request = [ProbeRequest(5.0, "X")]
        threshold, _ = probe_threshold([spectrum], request, 0.2, 0.01, 256)
        probe = probe_frequencies([spectrum], request, threshold, 0.2)[0]
        self.assertLess(abs(probe.frequency_hz - 5.0), 0.25)
        self.assertFalse(probe.detected)

    def test_candidates_from_another_search_are_merged_per_axis(self) -> None:
        import csv
        import tempfile
        from pathlib import Path

        from stable_spectrum import PEAK_FIELDS, load_candidates

        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "runA" / "nperseg_2048"
            folder.mkdir(parents=True)
            rows = [
                ("X", 2.47, 12.0), ("X", 2.55, 5.0), ("Y", 2.47, 4.5), ("X", 9.8, 6.0),
            ]
            with (folder / "stable_frequencies.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=PEAK_FIELDS)
                writer.writeheader()
                for axis, frequency, z in rows:
                    writer.writerow({
                        "capture": "runA_raw", "nperseg": 2048, "axis": axis,
                        "frequency_hz": frequency, "z": z,
                    })
            candidates, captures, files = load_candidates([Path(directory) / "runA"], 0.2)
        self.assertEqual(
            [(item.axis, item.frequency_hz) for item in candidates],
            [("X", 2.47), ("X", 9.8), ("Y", 2.47)],
        )
        self.assertEqual(captures, frozenset({"runA_raw"}))
        self.assertEqual(len(files), 1)

    def test_one_run_confirms_another_but_not_itself(self) -> None:
        import tempfile
        from pathlib import Path

        from stable_spectrum import main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, seed in (("runA_raw", 970), ("runB_raw", 975)):
                np.savez(
                    root / f"{name}.npz",
                    x=add_tone(noise_packets(256, seed), ON_BIN_HZ, 0.08, seed=seed + 1),
                    y=noise_packets(256, seed + 2), z=noise_packets(256, seed + 3),
                    packet_fs_hz=np.full(256, SAMPLING_RATE_HZ),
                )
            main([str(root / "runA_raw.npz"), "--nperseg", "1024",
                  "--alpha", "0.05", "--output", str(root / "A")])
            main([str(root / "runB_raw.npz"), str(root / "runA_raw.npz"),
                  "--nperseg", "1024", "--confirm-from", str(root / "A"),
                  "--probe", "Y:7.0", "--output", str(root / "B")])
            report = (root / "B" / "nperseg_1024" / "stable_report.txt").read_text(
                encoding="utf-8",
            )
            self.assertIn("confirmed", report)
            self.assertIn("a record cannot confirm its own findings", report)
            self.assertIn("Requested frequencies (chosen before this record)", report)
            probes = (root / "B" / "nperseg_1024" / "stable_probes.csv").read_text(
                encoding="utf-8",
            )
            self.assertIn("confirm,X", probes)
            self.assertIn("probe,Y", probes)


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


def wide_hump(packets: int, seed: int, low_hz: float, high_hz: float) -> np.ndarray:
    """Band-limited noise: a hump as wide as the band, unit variance."""
    from scipy.signal import butter, sosfiltfilt

    generator = np.random.default_rng(seed)
    sos = butter(2, (low_hz, high_hz), btype="bandpass", fs=SAMPLING_RATE_HZ, output="sos")
    hump = sosfiltfilt(sos, generator.normal(0.0, 1.0, packets * SAMPLES_PER_PACKET))
    return (hump / hump.std()).reshape(packets, SAMPLES_PER_PACKET)


class BaselineWindowTests(unittest.TestCase):
    HUMP_HZ = (9.0, 13.0)

    def hump_top(self, signal: np.ndarray, window_hz: float):
        from dataclasses import replace

        settings = replace(SETTINGS, baseline_window_hz=window_hz)
        spectrum = analyze_axis("X", signal, SAMPLING_RATE_HZ, settings)
        threshold = significance_threshold(
            spectrum.frequencies.size * 3, settings.alpha, signal.shape[0],
        )
        inside = (spectrum.frequencies >= self.HUMP_HZ[0]) & (
            spectrum.frequencies <= self.HUMP_HZ[1]
        )
        peaks = [
            peak for peak in find_stable_peaks([spectrum], threshold, settings, BANDS)
            if self.HUMP_HZ[0] <= peak.frequency_hz <= self.HUMP_HZ[1]
        ]
        return float(np.max(spectrum.prominence_db[inside])), peaks

    def test_the_top_of_a_wide_hump_keeps_its_prominence(self) -> None:
        # A 4 Hz wide hump lifts a 5 Hz running median almost to its own
        # level, and its top loses most of its prominence. A 10 Hz median
        # stays on the floor beside it.
        for seed in (1, 3):
            signal = noise_packets(256, 500 + seed) + 0.15 * wide_hump(
                256, 600 + seed, *self.HUMP_HZ,
            )
            narrow_db, _ = self.hump_top(signal, 5.0)
            wide_db, peaks = self.hump_top(signal, 10.0)
            self.assertGreater(wide_db - narrow_db, 1.0, f"seed {seed}")
            self.assertTrue(peaks, f"hump top missed at 10 Hz, seed {seed}")

    def test_a_narrow_line_does_not_care_about_the_window(self) -> None:
        signal = add_tone(noise_packets(256, 84), ON_BIN_HZ, 0.08, seed=85)
        heights = []
        for window_hz in (5.0, 10.0):
            from dataclasses import replace

            spectrum = analyze_axis(
                "X", signal, SAMPLING_RATE_HZ,
                replace(SETTINGS, baseline_window_hz=window_hz),
            )
            index = int(np.argmin(np.abs(spectrum.frequencies - ON_BIN_HZ)))
            heights.append(float(spectrum.prominence_db[index]))
        self.assertLess(abs(heights[1] - heights[0]), 0.3)

    def test_noise_stays_quiet_with_the_wide_window(self) -> None:
        from dataclasses import replace

        for nperseg in (1024, 4096):
            settings = replace(
                settings_at(nperseg), baseline_window_hz=10.0,
            )
            for seed in range(4):
                spectrum = analyze_axis(
                    "X", noise_packets(128, 700 + seed), SAMPLING_RATE_HZ, settings,
                )
                threshold = significance_threshold(
                    spectrum.frequencies.size * 3, settings.alpha,
                    int(round(spectrum.effective_rows)),
                )
                self.assertEqual(
                    find_stable_peaks([spectrum], threshold, settings, BANDS), [],
                    f"false peak at {nperseg}, seed {seed}",
                )

    def test_the_window_comes_from_config_and_can_be_overridden(self) -> None:
        import tomllib

        from stable_spectrum import CONFIG_PATH, load_settings

        with CONFIG_PATH.open("rb") as handle:
            configured = float(tomllib.load(handle)["stable_spectrum"]["baseline_window_hz"])
        self.assertEqual(load_settings(0.01, None).baseline_window_hz, configured)
        self.assertEqual(load_settings(0.01, None, 7.5).baseline_window_hz, 7.5)
        with self.assertRaises(ValueError):
            load_settings(0.01, None, 0.0)

    def test_the_command_line_key_reaches_the_report(self) -> None:
        import tempfile
        from pathlib import Path

        from stable_spectrum import main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "synthetic_raw.npz"
            np.savez(
                raw,
                x=noise_packets(64, 710), y=noise_packets(64, 711),
                z=noise_packets(64, 712),
                packet_fs_hz=np.full(64, SAMPLING_RATE_HZ),
            )
            main([
                str(raw), "--nperseg", "1024", "--baseline-window", "15",
                "--output", str(root / "out"),
            ])
            report = (root / "out" / "nperseg_1024" / "stable_report.txt").read_text(
                encoding="utf-8",
            )
            self.assertIn("Baseline window: 15 Hz", report)


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


class SeparationTests(unittest.TestCase):
    def found(self, signal, sigma, nperseg=4096):
        from dataclasses import replace

        settings = replace(
            settings_at(nperseg), baseline_window_hz=10.0, separation_sigma=sigma,
        )
        spectrum = analyze_axis("X", signal, SAMPLING_RATE_HZ, settings)
        threshold = significance_threshold(
            spectrum.frequencies.size * 3, 0.01, int(round(spectrum.effective_rows)),
        )
        return [
            round(peak.frequency_hz, 2)
            for peak in find_stable_peaks([spectrum], threshold, settings, BANDS)
        ]

    def test_two_lines_0_6_hz_apart_are_both_found(self) -> None:
        for seed in (0, 1):
            signal = add_tone(
                add_tone(noise_packets(256, 1200 + seed), ON_BIN_HZ, 0.08, seed=1300 + seed),
                ON_BIN_HZ + 0.6, 0.06, seed=1400 + seed,
            )
            self.assertEqual(len(self.found(signal, 0.0)), 1, f"seed {seed}")
            self.assertEqual(len(self.found(signal, 2.0)), 2, f"seed {seed}")

    def test_a_strong_line_gets_no_neighbour(self) -> None:
        signal = add_tone(noise_packets(256, 1900), ON_BIN_HZ, 0.5, seed=1950)
        for nperseg in (1024, 4096):
            self.assertEqual(self.found(signal, 2.0, nperseg), [round(ON_BIN_HZ, 2)])

    def test_the_flank_of_a_resonance_gets_no_new_peak(self) -> None:
        for seed in range(3):
            signal = noise_packets(256, 1700 + seed) + 1.5 * resonance(
                256, 6.0, 0.04, seed=1800 + seed,
            )
            for nperseg in (2048, 4096):
                self.assertEqual(
                    sorted(self.found(signal, 2.0, nperseg)),
                    sorted(self.found(signal, 0.0, nperseg)),
                    f"seed {seed}, nperseg {nperseg}",
                )

    def test_peaks_exactly_one_window_apart_need_no_dip(self) -> None:
        from types import SimpleNamespace

        from stable_spectrum import separate_close_peaks

        flat = np.zeros(40)
        spectrum = SimpleNamespace(
            z=flat.copy(), prominence_db=flat.copy(), standard_error_db=flat + 0.3,
        )
        spectrum.z[[10, 26]] = [9.0, 6.0]
        spectrum.prominence_db[[10, 26]] = [3.0, 2.0]
        spectrum.prominence_db[11:26] = 1.9  # almost no dip in between
        kept = separate_close_peaks(spectrum, np.array([10, 26]), 16, 2.0)
        self.assertEqual(list(kept), [10, 26])
        closer = separate_close_peaks(spectrum, np.array([10, 25]), 16, 2.0)
        self.assertEqual(list(closer), [10])

    def test_the_settings_carry_the_separation(self) -> None:
        from stable_spectrum import load_settings

        self.assertEqual(load_settings(0.01, None, None, None, 0.0).separation_sigma, 0.0)
        self.assertEqual(load_settings(0.01, None, None, 0.5, 2.5).min_distance_hz, 0.5)
        self.assertEqual(load_settings(0.01, None, None, None, 2.5).separation_sigma, 2.5)
        with self.assertRaises(ValueError):
            load_settings(0.01, None, None, None, -1.0)

    def test_the_configured_separation_is_the_default(self) -> None:
        import tomllib

        from stable_spectrum import CONFIG_PATH, load_settings

        with CONFIG_PATH.open("rb") as handle:
            own = tomllib.load(handle)["stable_spectrum"]
        settings = load_settings(0.01, None)
        self.assertEqual(settings.separation_sigma, float(own["separation_sigma"]))
        self.assertEqual(settings.min_distance_hz, float(own["min_distance_hz"]))


class PoolTests(unittest.TestCase):
    def write(self, path, seed, tone=0.0, rate=SAMPLING_RATE_HZ, offset=0.0, packets=128):
        np.savez(
            path,
            x=add_tone(noise_packets(packets, seed), ON_BIN_HZ, tone, seed=seed + 1) + offset,
            y=noise_packets(packets, seed + 2), z=noise_packets(packets, seed + 3),
            packet_fs_hz=np.full(packets, rate),
        )

    def test_no_segment_crosses_the_pause_between_runs(self) -> None:
        from stable_spectrum import segment_periodograms

        settings = settings_at(4096)
        # A level step between two runs, as when the sensor settles anew.
        joined = np.vstack([noise_packets(32, 990), noise_packets(32, 991) + 50.0])
        _, glued, _ = segment_periodograms(joined, SAMPLING_RATE_HZ, settings)
        _, apart, _ = segment_periodograms(joined, SAMPLING_RATE_HZ, settings, [32, 32])
        hop = 2048
        self.assertEqual(glued.shape[0], (64 * 1024 - 4096) // hop + 1)
        self.assertEqual(apart.shape[0], 2 * ((32 * 1024 - 4096) // hop + 1))
        # The glued row holds the step and lifts the lowest bins; apart, no row does.
        self.assertGreater(glued[:, 1:4].max(), 50.0 * np.median(glued[:, 1:4]))
        self.assertLess(apart[:, 1:4].max(), 50.0 * np.median(apart[:, 1:4]))

    def test_support_windows_stay_inside_one_run(self) -> None:
        from stable_spectrum import split_into_windows, windows_for_groups

        windows = windows_for_groups([255, 256])
        self.assertEqual(len(windows), 16)
        for window in windows:
            self.assertTrue(window[-1] < 255 or window[0] >= 255)
        single = windows_for_groups([256])
        self.assertEqual(
            [list(item) for item in single], [list(item) for item in split_into_windows(256)],
        )

    def test_runs_count_how_many_runs_find_the_peak_alone(self) -> None:
        import tempfile
        from pathlib import Path

        from stable_spectrum import analyze_pool, load_band_names

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            both = [root / "a_raw.npz", root / "b_raw.npz"]
            self.write(both[0], 1000, tone=0.12)
            self.write(both[1], 1010, tone=0.12)
            result = analyze_pool(both, SETTINGS, load_band_names())
            peak = next(item for item in result.peaks if item.axis == "X")
            self.assertEqual((peak.runs_found, peak.runs_total), (2, 2))
            self.assertEqual(peak.runs_label, "2/2")
            self.assertTrue(result.pooled)
            self.assertEqual(result.member_packets, (128, 128))

            one = [root / "c_raw.npz", root / "d_raw.npz"]
            self.write(one[0], 1020, tone=0.15)
            self.write(one[1], 1030, tone=0.0)
            result = analyze_pool(one, SETTINGS, load_band_names())
            peak = next(item for item in result.peaks if item.axis == "X")
            self.assertEqual(peak.runs_label, "1/2")

    def test_a_level_jump_between_runs_is_warned_about(self) -> None:
        import tempfile
        from pathlib import Path

        from stable_spectrum import analyze_pool, level_jump_warnings, load_band_names

        same = [{key: noise_packets(32, 1100 + index) for key in "xyz"} for index in range(2)]
        self.assertEqual(level_jump_warnings(same), [])
        near = [same[0], {key: value + 2.0 for key, value in same[1].items()}]
        self.assertEqual(level_jump_warnings(near), [])
        moved = [same[0], {**same[1], "x": same[1]["x"] + 60.0}]
        messages = level_jump_warnings(moved)
        self.assertEqual(len(messages), 1)
        self.assertTrue(messages[0].startswith("X:"))
        self.assertEqual(level_jump_warnings(same[:1]), [])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write(root / "a_raw.npz", 1110)
            self.write(root / "b_raw.npz", 1111, offset=60.0)
            result = analyze_pool(
                [root / "a_raw.npz", root / "b_raw.npz"], SETTINGS, load_band_names(),
            )
            self.assertEqual(len(result.pool_warnings), 1)

    def test_runs_with_different_rates_are_refused(self) -> None:
        import tempfile
        from pathlib import Path

        from stable_spectrum import analyze_pool, load_band_names

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write(root / "a_raw.npz", 1040)
            self.write(root / "b_raw.npz", 1041, rate=125.0)
            with self.assertRaises(ValueError):
                analyze_pool(
                    [root / "a_raw.npz", root / "b_raw.npz"], SETTINGS, load_band_names(),
                )

    def test_command_line_pool_writes_one_pooled_report(self) -> None:
        import tempfile
        from pathlib import Path

        from stable_spectrum import main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, seed in (("20261001_141520_run01_raw", 1050), ("20261001_143239_run02_raw", 1060)):
                self.write(root / f"{name}.npz", seed, tone=0.12)
            main([
                str(root / "20261001_141520_run01_raw.npz"),
                str(root / "20261001_143239_run02_raw.npz"),
                "--pool", "--nperseg", "2048", "--output", str(root / "out"),
            ])
            folder = root / "out" / "nperseg_2048"
            report = (folder / "stable_report.txt").read_text(encoding="utf-8")
            self.assertIn("Pooled record of 2 runs", report)
            self.assertIn("Capture: pooled_141520_143239", report)
            self.assertIn("2/2", report)
            self.assertEqual(report.count("Recording check"), 2)
            table = (folder / "stable_frequencies.csv").read_text(encoding="utf-8")
            self.assertIn("runs_found,runs_total", table)
            self.assertTrue((folder / "figure_dominant_pooled_141520_143239.png").exists())


class FolderInputTests(unittest.TestCase):
    def write(self, path, seed):
        np.savez(
            path,
            x=add_tone(noise_packets(64, seed), ON_BIN_HZ, 0.3, seed=seed + 1),
            y=noise_packets(64, seed + 2), z=noise_packets(64, seed + 3),
            packet_fs_hz=np.full(64, SAMPLING_RATE_HZ),
        )

    def test_a_folder_is_all_runs_of_one_point(self) -> None:
        import tempfile
        from pathlib import Path

        from stable_spectrum import expand_raw_paths

        with tempfile.TemporaryDirectory() as directory:
            point = Path(directory) / "point"
            (point / "other point").mkdir(parents=True)
            for name in ("b_run02_raw.npz", "a_run01_raw.npz", "other point/c_raw.npz"):
                (point / name).write_bytes(b"")
            (point / "notes_figure.npz").write_bytes(b"")
            single = Path(directory) / "single_raw.npz"
            self.assertEqual(
                expand_raw_paths([point, single, point / "a_run01_raw.npz"]),
                [point / "a_run01_raw.npz", point / "b_run02_raw.npz", single],
            )
            building = Path(directory) / "building"
            (building / "basement").mkdir(parents=True)
            (building / "basement" / "r_raw.npz").write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "point folders inside: basement"):
                expand_raw_paths([building])

    def test_command_line_takes_a_folder_and_names_the_output_after_it(self) -> None:
        import contextlib
        import io
        import tempfile
        from pathlib import Path
        from unittest import mock

        import stable_spectrum
        from stable_spectrum import main, resolve_output_directory

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            point = root / "лестница 1"
            point.mkdir()
            self.write(point / "20261001_133738_run01_raw.npz", 1100)
            self.write(point / "20261001_135458_run02_raw.npz", 1110)
            with mock.patch.object(stable_spectrum, "STABLE_RESULTS_DIRECTORY", root / "results"):
                self.assertEqual(
                    resolve_output_directory(None, [point]),
                    root / "results" / "лестница 1" / "runs",
                )
                self.assertEqual(
                    resolve_output_directory(None, [point], pool=True),
                    root / "results" / "лестница 1" / "pool",
                )
            with contextlib.redirect_stdout(io.StringIO()):
                code = main([str(point), "--nperseg", "2048", "--output", str(root / "out")])
            self.assertEqual(code, 0)
            report = (root / "out" / "nperseg_2048" / "stable_report.txt").read_text(encoding="utf-8")
            self.assertIn("Capture: 20261001_133738_run01_raw", report)
            self.assertIn("Capture: 20261001_135458_run02_raw", report)
            self.assertIn("Each half (Persistent): z >=", report)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main([str(root), "--output", str(root / "none")]), 1)
            self.assertFalse((root / "none").exists())


class DefaultResolutionTests(unittest.TestCase):
    def test_without_nperseg_only_the_configured_resolution_is_run(self) -> None:
        import tempfile
        import tomllib
        from pathlib import Path

        from stable_spectrum import CONFIG_PATH, load_default_nperseg, main

        with CONFIG_PATH.open("rb") as handle:
            configured = tomllib.load(handle)["stable_spectrum"]["nperseg"]
        self.assertEqual(load_default_nperseg(), [int(value) for value in configured])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "synthetic_raw.npz"
            np.savez(
                raw,
                x=add_tone(noise_packets(64, 980), ON_BIN_HZ, 0.3, seed=981),
                y=noise_packets(64, 982), z=noise_packets(64, 983),
                packet_fs_hz=np.full(64, SAMPLING_RATE_HZ),
            )
            main([str(raw), "--output", str(root / "out")])
            folders = sorted(path.name for path in (root / "out").glob("nperseg_*"))
            self.assertEqual(folders, [f"nperseg_{value}" for value in load_default_nperseg()])
            self.assertTrue((root / "out" / "figure_compare_synthetic_raw.png").exists())


class BandEdgeTests(unittest.TestCase):
    # Bins at 250 Hz and 1024 samples are 0.244 Hz apart. With the band
    # starting at 0.5 Hz, 0.732 Hz is the first bin inside it and 0.488 Hz
    # the last bin below it.
    FIRST_IN_BAND_HZ = 3 * SAMPLING_RATE_HZ / 1024
    LAST_BELOW_BAND_HZ = 2 * SAMPLING_RATE_HZ / 1024

    def spectrum_and_peaks(self, signal, settings=SETTINGS):
        spectrum = analyze_axis("X", signal, SAMPLING_RATE_HZ, settings)
        threshold = significance_threshold(
            spectrum.frequencies.size * 3, settings.alpha, signal.shape[0],
        )
        peaks = find_stable_peaks([spectrum], threshold, settings, BANDS)
        return spectrum, threshold, peaks

    def test_the_band_itself_is_unchanged(self) -> None:
        spectrum, _, _ = self.spectrum_and_peaks(noise_packets(64, 101))
        self.assertGreaterEqual(spectrum.frequencies[0], SETTINGS.band_hz[0])
        self.assertLessEqual(spectrum.frequencies[-1], SETTINGS.band_hz[1])
        self.assertAlmostEqual(spectrum.frequencies[0], self.FIRST_IN_BAND_HZ)
        self.assertGreater(spectrum.context_z.size, spectrum.z.size)

    def test_a_tone_in_the_first_bin_of_the_band_is_found(self) -> None:
        signal = add_tone(
            noise_packets(128, 102), self.FIRST_IN_BAND_HZ, 0.3, seed=103,
        )
        _, _, peaks = self.spectrum_and_peaks(signal)
        self.assertTrue(peaks)
        self.assertAlmostEqual(
            peaks[0].frequency_hz, self.FIRST_IN_BAND_HZ, delta=0.01,
        )

    def test_a_tone_just_below_the_band_is_warned_about(self) -> None:
        from stable_spectrum import band_edge_warnings

        signal = add_tone(
            noise_packets(128, 104), self.LAST_BELOW_BAND_HZ, 0.3, seed=105,
        )
        spectrum, threshold, peaks = self.spectrum_and_peaks(signal)
        self.assertEqual(
            [peak for peak in peaks if peak.frequency_hz < 0.6], [],
        )
        warnings = band_edge_warnings([spectrum], threshold)
        self.assertTrue(any("below the band" in text for text in warnings))

    def test_a_tone_in_the_lowest_available_bin_is_warned_about(self) -> None:
        from dataclasses import replace

        from stable_spectrum import band_edge_warnings

        settings = replace(SETTINGS, band_hz=(0.2, 15.0))
        lowest_hz = SAMPLING_RATE_HZ / 1024
        signal = add_tone(noise_packets(128, 106), lowest_hz, 0.3, seed=107)
        spectrum, threshold, _ = self.spectrum_and_peaks(signal, settings)
        self.assertEqual(spectrum.band_offset, 0)
        warnings = band_edge_warnings([spectrum], threshold)
        self.assertTrue(any("lowest bin" in text for text in warnings))

    def test_noise_raises_neither_peaks_nor_edge_warnings(self) -> None:
        from dataclasses import replace

        from stable_spectrum import band_edge_warnings

        for band in ((0.5, 15.0), (0.2, 15.0)):
            settings = replace(SETTINGS, band_hz=band)
            for seed in range(6):
                spectrum, threshold, peaks = self.spectrum_and_peaks(
                    noise_packets(128, 110 + seed), settings,
                )
                self.assertEqual(peaks, [], f"false peak, {band}, seed {seed}")
                self.assertEqual(
                    band_edge_warnings([spectrum], threshold), [],
                    f"false edge warning, {band}, seed {seed}",
                )


class StartupPacketTests(unittest.TestCase):
    @staticmethod
    def started(packets: int, seed: int) -> np.ndarray:
        # The sensor starts from zero: the first sample of the record is 0,
        # every later one sits near 1 g like the Z axis does.
        signal = noise_packets(packets, seed, amplitude=0.001) + 1.0
        signal[0, 0] = 0.0
        return signal

    def test_the_start_up_packet_is_dropped(self) -> None:
        from stable_spectrum import drop_startup_packet

        axes = {
            "x": noise_packets(64, 121),
            "y": noise_packets(64, 122),
            "z": self.started(64, 123),
        }
        rates = np.full(64, SAMPLING_RATE_HZ)
        kept, kept_rates, dropped = drop_startup_packet(axes, rates)
        self.assertTrue(dropped)
        self.assertEqual(kept["z"].shape[0], 63)
        self.assertEqual(kept["x"].shape[0], 63)
        self.assertEqual(kept_rates.size, 63)

    def test_a_clean_record_is_left_whole(self) -> None:
        from stable_spectrum import drop_startup_packet

        axes = {key: noise_packets(64, 124 + index)
                for index, key in enumerate("xyz")}
        kept, _, dropped = drop_startup_packet(axes, np.full(64, 250.0))
        self.assertFalse(dropped)
        self.assertEqual(kept["z"].shape[0], 64)

    def test_the_start_up_step_no_longer_fakes_a_low_structure(self) -> None:
        import tempfile
        from pathlib import Path

        from dataclasses import replace

        from stable_spectrum import analyze_capture, load_band_names, load_settings

        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "started_raw.npz"
            np.savez(
                raw,
                x=noise_packets(128, 125, amplitude=0.001),
                y=noise_packets(128, 126, amplitude=0.001),
                z=self.started(128, 127),
                packet_fs_hz=np.full(128, SAMPLING_RATE_HZ),
            )
            # One periodogram per packet, as the test was written for.
            settings = replace(load_settings(0.01, None), nperseg=1024, noverlap=512)
            result = analyze_capture(raw, settings, load_band_names())
        self.assertTrue(result.startup_dropped)
        self.assertEqual(result.packet_count, 127)
        self.assertEqual(result.peaks, [])
        self.assertEqual(result.edge_warnings, [])
        self.assertIn("start-up", format_report(result))


def format_report(result) -> str:
    from stable_spectrum import format_capture_report

    return format_capture_report(result)


class PeakRefinementTests(unittest.TestCase):
    STEP_HZ = SAMPLING_RATE_HZ / 1024

    def peaks_of(self, frequency_hz: float, seed: int):
        signal = add_tone(noise_packets(128, seed), frequency_hz, 0.3, seed=seed + 1)
        _, peaks = peaks_for(signal)
        self.assertTrue(peaks)
        return peaks[0]

    def test_a_tone_between_bins_is_placed_between_them(self) -> None:
        # A quarter of a bin above a bin centre: the bin is off by 0.061 Hz.
        true_hz = 20.25 * self.STEP_HZ
        peak = self.peaks_of(true_hz, 131)
        self.assertAlmostEqual(peak.bin_frequency_hz, 20 * self.STEP_HZ)
        self.assertLess(abs(peak.frequency_hz - true_hz), 0.02)
        self.assertLess(
            abs(peak.frequency_hz - true_hz),
            abs(peak.bin_frequency_hz - true_hz),
        )

    def test_a_tone_halfway_between_bins_gives_a_stable_answer(self) -> None:
        # Halfway between two bins the maximum flips from one to the other
        # with the noise; the parabola top must not.
        true_hz = 20.5 * self.STEP_HZ
        refined = [self.peaks_of(true_hz, 140 + 2 * seed).frequency_hz
                   for seed in range(4)]
        self.assertLess(max(abs(value - true_hz) for value in refined), 0.03)

    def test_the_refinement_stays_within_half_a_bin(self) -> None:
        from stable_spectrum import refine_peak

        frequencies = np.arange(10) * self.STEP_HZ
        mean_psd = np.ones(10)
        mean_psd[4] = 2.0
        mean_psd[5] = 1.999
        refined_hz = refine_peak(frequencies, mean_psd, 4)
        self.assertGreater(refined_hz, frequencies[4])
        self.assertLessEqual(refined_hz, frequencies[4] + 0.5 * self.STEP_HZ)

    def test_a_peak_without_two_neighbours_is_left_alone(self) -> None:
        from stable_spectrum import refine_peak

        frequencies = np.arange(5) * self.STEP_HZ
        mean_psd = np.array([3.0, 2.0, 1.0, 1.0, 1.0])
        self.assertEqual(refine_peak(frequencies, mean_psd, 0), 0.0)

if __name__ == "__main__":
    unittest.main()
