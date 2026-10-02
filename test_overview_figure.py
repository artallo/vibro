"""Unit tests for the folder overview figure.

Run with: python -m unittest test_overview_figure
"""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from overview_figure import (
    build_overview,
    find_dips,
    load_folder,
    main as overview_main,
    save_overview,
)
from stable_spectrum import significance_threshold


def main(arguments: list[str]) -> int:
    with contextlib.redirect_stdout(io.StringIO()):
        return overview_main(arguments)

RATE_HZ = 250.0
TONE_HZ = 20 * RATE_HZ / 1024


def write_capture(
    path: Path,
    seed: int,
    packets: int = 64,
    rate_hz: float = RATE_HZ,
    tone: float = 0.0,
) -> None:
    generator = np.random.default_rng(seed)
    time = np.arange(packets * 1024) / rate_hz
    x = generator.normal(0.0, 1.0, packets * 1024)
    x += tone * np.sin(2.0 * np.pi * TONE_HZ * time)
    np.savez(
        path,
        x=x.reshape(packets, 1024),
        y=generator.normal(0.0, 1.0, (packets, 1024)),
        z=generator.normal(0.0, 1.0, (packets, 1024)),
        packet_fs_hz=np.full(packets, rate_hz),
    )


class OverviewTests(unittest.TestCase):
    def test_all_captures_are_pooled_and_a_tone_is_found(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for seed in range(3):
                write_capture(folder / f"run{seed}_raw.npz", seed, tone=0.15)
            (folder / "broken.npz").write_bytes(b"not an archive")
            np.savez(folder / "other.npz", values=np.arange(3))
            captures = load_folder(folder)
            self.assertEqual(len(captures), 3)
            overview = build_overview(captures)
            self.assertEqual(overview.pooled[0].row_count, 192)
            self.assertTrue(overview.pooled_peaks)
            self.assertEqual(overview.pooled_peaks[0].axis, "X")
            self.assertAlmostEqual(
                overview.pooled_peaks[0].frequency_hz, TONE_HZ, delta=0.25,
            )
            self.assertEqual(len(overview.full_band["X"]), 3)
            output = folder / "out" / "figure_overview.png"
            save_overview(output, overview)
            self.assertTrue(output.exists())

    def test_noise_only_folder_reports_nothing_in_band(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for seed in range(2):
                write_capture(folder / f"run{seed}_raw.npz", 10 + seed)
            overview = build_overview(load_folder(folder))
            self.assertEqual(overview.pooled_peaks, [])

    def test_longer_segments_find_the_tone_at_their_own_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for seed in range(3):
                write_capture(folder / f"run{seed}_raw.npz", seed, tone=0.15)
            captures = load_folder(folder)
            for nperseg in (2048, 4096):
                overview = build_overview(captures, nperseg=nperseg)
                self.assertEqual(overview.nperseg, nperseg)
                self.assertAlmostEqual(
                    overview.bin_width_hz, RATE_HZ / nperseg, places=6,
                )
                # 192 packets in one row, half-overlapping segments.
                hop = nperseg // 2
                self.assertEqual(
                    overview.segment_count, (192 * 1024 - nperseg) // hop + 1,
                )
                self.assertLess(
                    overview.effective_segments, overview.segment_count,
                )
                bins = sum(item.frequencies.size for item in overview.pooled)
                self.assertAlmostEqual(
                    overview.pooled_threshold,
                    significance_threshold(
                        bins, 0.01, int(round(overview.effective_segments)),
                    ),
                )
                self.assertTrue(overview.pooled_peaks)
                self.assertEqual(overview.pooled_peaks[0].axis, "X")
                self.assertAlmostEqual(
                    overview.pooled_peaks[0].frequency_hz, TONE_HZ,
                    delta=RATE_HZ / nperseg,
                )

    def test_looser_alpha_keeps_the_default_threshold_for_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            write_capture(folder / "run0_raw.npz", 3)
            captures = load_folder(folder)
            strict = build_overview(captures)
            loose = build_overview(captures, alpha=0.05)
            self.assertAlmostEqual(strict.pooled_threshold, strict.reference_threshold)
            self.assertLess(loose.pooled_threshold, loose.reference_threshold)
            self.assertAlmostEqual(loose.reference_threshold, strict.pooled_threshold)
            self.assertFalse(strict.searching)
            self.assertTrue(loose.searching)
            for peak in strict.pooled_peaks:
                self.assertTrue(strict.passes_reference(peak))
            output = folder / "out" / "figure_overview.png"
            save_overview(output, loose)
            self.assertTrue(output.exists())

    def test_dips_are_the_peak_search_turned_upside_down(self) -> None:
        frequencies = np.arange(40) * 0.25
        z = np.zeros(40)
        z[10] = -6.0
        z[25:31] = -5.0
        z[28] = -5.5
        z[35] = 6.0
        spectrum = SimpleNamespace(
            axis="Y", frequencies=frequencies, z=z,
            context_z=None, band_offset=0,
        )
        settings = SimpleNamespace(min_distance_hz=1.0)
        dips = find_dips([spectrum], 4.0, settings)
        self.assertEqual([dip.frequency_hz for dip in dips], [2.5, 7.0])
        self.assertTrue(dips[0].narrow)
        self.assertEqual(dips[1].width_bins, 6)
        self.assertFalse(dips[1].narrow)

    def test_each_segment_length_gets_its_own_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "captures"
            folder.mkdir()
            for seed in range(2):
                write_capture(folder / f"run{seed}_raw.npz", 20 + seed, packets=16)
            output = Path(directory) / "out"
            code = main([
                str(folder), "--output", str(output),
                "--nperseg", "1024", "4096", "1048576", "--alpha", "0.05",
            ])
            self.assertEqual(code, 0)
            for nperseg in (1024, 4096):
                resolution = output / f"nperseg_{nperseg}"
                self.assertTrue((resolution / "figure_overview.png").exists())
                report = (resolution / "overview.txt").read_text(encoding="utf-8")
                self.assertIn(f"nperseg {nperseg}", report)
                self.assertIn("at alpha 0.01", report)
                self.assertIn("Dips below", report)
            self.assertFalse((output / "nperseg_1048576").exists())

    def test_mixed_sampling_rates_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            write_capture(folder / "a_raw.npz", 1, rate_hz=250.0)
            write_capture(folder / "b_raw.npz", 2, rate_hz=125.0)
            with self.assertRaises(ValueError):
                load_folder(folder)

    def test_empty_folder_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                load_folder(Path(directory))


if __name__ == "__main__":
    unittest.main()
