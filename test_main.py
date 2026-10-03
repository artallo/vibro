"""Tests of main.py, the recording program and its old detector.

main.py cannot be imported: on import it reads the command line and, for a
live run, opens the serial port. The tests run it as a separate process.

Run with: python -m unittest test_main
"""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from false_alarm_check import save_capture

MAIN_PATH = Path(__file__).with_name("main.py")
RATE_HZ = 250.0


def run_main(*arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(MAIN_PATH), *arguments],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )


class RefusalTests(unittest.TestCase):
    def refused(self, *arguments: str) -> str:
        completed = run_main(*arguments)
        self.assertNotEqual(completed.returncode, 0)
        return completed.stdout + completed.stderr

    def test_layouts_the_mode_has_no_threshold_for_are_refused(self) -> None:
        # A live run is refused before the serial port is opened.
        self.assertIn(
            "layout 8x5 has no Med.Prom threshold for nperseg 2048",
            self.refused("--min-recommended-sessions", "5", "--nperseg", "2048"),
        )
        self.assertIn(
            "replay layout 4x16 has no Med.Prom threshold for nperseg 2048",
            self.refused("--replay", "missing_raw.npz", "--virtual-mode", "4x16", "--nperseg", "2048"),
        )
        self.assertIn(
            "--nperseg 4096 has no [old_detector.nperseg_4096] section",
            self.refused("--replay", "missing_raw.npz", "--nperseg", "4096"),
        )


class RefinedFrequencyTests(unittest.TestCase):
    def test_a_line_between_bins_is_reported_at_its_own_frequency(self) -> None:
        # At 250 Hz and nperseg 2048 the bins are 0.122 Hz apart: 3.00 Hz lies
        # between 2.930 and 3.052 Hz, 0.05 Hz from the nearer bin centre.
        tone_hz = 3.0
        bin_width = RATE_HZ / 2048
        nearest_bin = round(tone_hz / bin_width) * bin_width
        self.assertGreater(abs(nearest_bin - tone_hz), 0.04)
        generator = np.random.default_rng(7)
        time = np.arange(64 * 1024) / RATE_HZ
        axes = {key: generator.normal(0.0, 1.0, 64 * 1024) for key in "xyz"}
        axes["y"] += 0.4 * np.sin(2.0 * np.pi * tone_hz * time)
        axes = {key: value.reshape(64, 1024) for key, value in axes.items()}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "tone_raw.npz"
            save_capture(raw, axes, RATE_HZ, "test")
            completed = run_main(
                "--replay", str(raw), "--nperseg", "2048", "--virtual-mode", "8x8",
                "--replay-root", str(root / "replay"),
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            folder = root / "replay" / "tone_raw" / "nperseg_2048"
            with (folder / "replay_regions.csv").open(encoding="utf-8", newline="") as file:
                rows = [row for row in csv.DictReader(file) if row["axis"] == "Y"]
            tone = min(rows, key=lambda row: abs(float(row["med_freq_hz"]) - tone_hz))
            self.assertLess(abs(float(tone["med_freq_hz"]) - tone_hz), 0.02)
            # The detector still decides on the bin, which is kept alongside.
            self.assertAlmostEqual(float(tone["med_freq_bin_hz"]), nearest_bin, delta=0.001)
            run = folder / "8x8" / "virtual_run01"
            self.assertIn(f"{float(tone['med_freq_hz']):.2f}", (run / "result.txt").read_text(encoding="utf-8"))
            self.assertTrue((run / "figure1.png").exists())
            self.assertTrue((run / "figure2.png").exists())


if __name__ == "__main__":
    unittest.main()
