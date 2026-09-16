"""Unit tests for the folder overview figure.

Run with: python -m unittest test_overview_figure
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from overview_figure import build_overview, load_folder, save_overview

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
