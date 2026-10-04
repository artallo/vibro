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


def run_main(*arguments: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(MAIN_PATH), *arguments],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=env,
    )


def write_tone_record(path: Path, tone_hz: float = 3.0, packets: int = 64) -> None:
    generator = np.random.default_rng(7)
    time = np.arange(packets * 1024) / RATE_HZ
    axes = {key: generator.normal(0.0, 1.0, packets * 1024) for key in "xyz"}
    axes["y"] += 0.4 * np.sin(2.0 * np.pi * tone_hz * time)
    axes = {key: value.reshape(packets, 1024) for key, value in axes.items()}
    save_capture(path, axes, RATE_HZ, "test")


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

    def test_show_needs_a_replay_of_one_layout(self) -> None:
        message = "--show needs --replay and --virtual-mode with one layout"
        for arguments in (
            ("--show",),
            ("--replay", "missing_raw.npz", "--show"),
            ("--replay", "missing_raw.npz", "--show", "--virtual-mode", "all"),
            ("--replay", "missing_raw.npz", "--show", "--virtual-mode", "8x8", "8x16"),
        ):
            self.assertIn(message, self.refused(*arguments))


class ShowTests(unittest.TestCase):
    def test_show_opens_the_window_and_writes_nothing(self) -> None:
        import os

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "tone_raw.npz"
            write_tone_record(raw, packets=128)
            # Without a display the window is a no-op; the analysis still runs.
            env = {**os.environ, "MPLBACKEND": "Agg"}
            completed = run_main(
                "--replay", str(raw), "--nperseg", "2048", "--virtual-mode", "8x8",
                "--show", "--replay-root", str(root / "replay"), env=env,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertFalse((root / "replay").exists())


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
            report = (run / "result.txt").read_text(encoding="utf-8")
            self.assertIn(f"{float(tone['med_freq_hz']):.2f}", report)
            self.assertTrue((run / "figure1.png").exists())
            self.assertTrue((run / "figure2.png").exists())
            # Candidates below the threshold: none on the axis with a trusted
            # region, at most two on each other axis, each at its own maximum.
            self.assertIn(
                "Candidates below the threshold — Y\nTrusted frequency regions present; no candidates.",
                report,
            )
            with (folder / "replay_candidates.csv").open(encoding="utf-8", newline="") as file:
                candidates = list(csv.DictReader(file))
            self.assertFalse([row for row in candidates if row["axis"] == "Y"])
            for axis in "XZ":
                picked = [row for row in candidates if row["axis"] == axis]
                self.assertLessEqual(len(picked), 2)
                for row in picked:
                    self.assertLessEqual(float(row["range_min_hz"]), float(row["med_freq_bin_hz"]))
                    self.assertLessEqual(float(row["med_freq_bin_hz"]), float(row["range_max_hz"]))
                    self.assertGreaterEqual(float(row["support_fraction"]), 0.5)
                    self.assertLess(float(row["med_prom_db"]), float(row["threshold_db"]))
                if len(picked) == 2:
                    self.assertEqual(
                        [row["picked_by"] for row in picked], ["Med.Prom", "support x Med.Prom"],
                    )
                    self.assertGreaterEqual(float(picked[0]["med_prom_db"]), float(picked[1]["med_prom_db"]))


if __name__ == "__main__":
    unittest.main()
