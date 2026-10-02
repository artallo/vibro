"""Unit tests for the control point runs.

Run with: python -m unittest test_control_run
"""

from __future__ import annotations

import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import control_run
from control_run import load_points, match_peaks, select_points

RATE_HZ = 250.0
TONE_HZ = 40 * RATE_HZ / 2048


def main(arguments: list[str]) -> int:
    with contextlib.redirect_stdout(io.StringIO()):
        return control_run.main(arguments)


def write_capture(path: Path, seed: int, tone: float = 0.0, rate_hz: float = RATE_HZ) -> None:
    generator = np.random.default_rng(seed)
    packets = 64
    time = np.arange(packets * 1024) / rate_hz
    x = generator.normal(0.0, 1.0, packets * 1024) + tone * np.sin(2.0 * np.pi * TONE_HZ * time)
    np.savez(
        path,
        x=x.reshape(packets, 1024),
        y=generator.normal(0.0, 1.0, (packets, 1024)),
        z=generator.normal(0.0, 1.0, (packets, 1024)),
        packet_fs_hz=np.full(packets, rate_hz),
    )


def write_points(path: Path, entries: list[tuple[str, Path, list[str]]]) -> None:
    lines = []
    for name, folder, probes in entries:
        lines += ["[[point]]", f'name = "{name}"', f'folder = "{folder.as_posix()}"']
        if probes:
            lines.append("probes = [" + ", ".join(f'"{probe}"' for probe in probes) + "]")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


class ControlRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.output = root / "control"
        self.tone = root / "tone"
        self.mixed = root / "mixed"
        self.tone.mkdir()
        self.mixed.mkdir()
        for seed in range(2):
            write_capture(self.tone / f"run{seed}_raw.npz", seed, tone=0.15)
        write_capture(self.mixed / "a_raw.npz", 5, rate_hz=250.0)
        write_capture(self.mixed / "b_raw.npz", 6, rate_hz=125.0)
        self.points = root / "points.toml"
        write_points(self.points, [
            ("tone", self.tone, [f"X:{TONE_HZ:.3f}"]),
            ("mixed", self.mixed, []),
        ])

    def tearDown(self) -> None:
        self.directory.cleanup()

    def run_variant(self, name: str, *extra: str) -> Path:
        code = main([
            name, "--points-file", str(self.points), "--output", str(self.output),
            "--nperseg", "2048", *extra,
        ])
        self.assertEqual(code, 0)
        return self.output / name

    def test_each_point_runs_alone_and_pooled_at_one_resolution(self) -> None:
        variant = self.run_variant("base")
        self.assertTrue((variant / "tone" / "runs" / "stable_frequencies.csv").exists())
        self.assertTrue((variant / "tone" / "pool" / "stable_frequencies.csv").exists())
        # One segment length: no nperseg_<n> folders and no comparison figures.
        self.assertFalse(list(variant.rglob("nperseg_*")))
        self.assertFalse(list(variant.rglob("figure_compare_*")))
        peaks = read_rows(variant / "control_peaks.csv")
        tone = [row for row in peaks if row["point"] == "tone" and row["axis"] == "X"]
        self.assertEqual({row["record"] for row in tone}, {"run0_raw", "run1_raw", "pool"})
        self.assertTrue(all(row["nperseg"] == "2048" for row in peaks))
        pooled = next(row for row in tone if row["record"] == "pool")
        self.assertEqual(pooled["runs"], "2/2")
        probes = read_rows(variant / "control_probes.csv")
        self.assertTrue(probes)
        self.assertTrue(all(row["detected"] == "1" for row in probes if row["point"] == "tone"))
        summary = (variant / "control_summary.txt").read_text(encoding="utf-8")
        self.assertIn("nperseg 2048", summary)
        # Runs at different sampling rates are analysed alone, not pooled.
        self.assertIn("different sampling rates", summary)

    def test_a_rerun_replaces_the_variant(self) -> None:
        variant = self.run_variant("base")
        stale = variant / "stale.txt"
        stale.write_text("old", encoding="utf-8")
        self.run_variant("base", "--points", "mixed")
        self.assertFalse(stale.exists())
        self.assertFalse((variant / "tone").exists())

    def test_comparison_lists_lost_new_and_kept_peaks(self) -> None:
        self.run_variant("before")
        # A threshold so strict that the tone is gone from single runs.
        self.run_variant("after", "--alpha", "1e-40")
        code = main(["--compare", "before", "after", "--output", str(self.output)])
        self.assertEqual(code, 0)
        text = (self.output / "compare_before_after.txt").read_text(encoding="utf-8")
        self.assertIn("Lost (in 'before' only):", text)
        lost = text.split("Lost (in 'before' only):")[1].split("New")[0]
        self.assertIn("run0_raw", lost)
        self.assertIn(f"X {TONE_HZ:.2f}", lost)

    def test_peaks_pair_on_the_same_record_and_axis_only(self) -> None:
        def peak(record: str, axis: str, frequency: float) -> dict:
            return {"point": "p", "record": record, "axis": axis,
                    "frequency_hz": frequency, "z": 5.0}

        before = [peak("r1", "X", 2.50), peak("r1", "Y", 2.90), peak("r2", "X", 2.50)]
        after = [peak("r1", "X", 2.58), peak("r1", "X", 2.45), peak("r1", "Y", 3.20)]
        pairs, lost, new = match_peaks(before, after, 0.2)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][1]["frequency_hz"], 2.45)
        self.assertEqual(
            {(row["record"], row["axis"]) for row in lost}, {("r1", "Y"), ("r2", "X")},
        )
        self.assertEqual({row["frequency_hz"] for row in new}, {2.58, 3.20})

    def test_several_configured_resolutions_are_refused(self) -> None:
        with mock.patch.object(control_run, "load_default_nperseg", return_value=[2048, 4096]):
            code = main(["base", "--points-file", str(self.points), "--output", str(self.output)])
        self.assertEqual(code, 1)
        self.assertFalse(self.output.exists())
        with mock.patch.object(control_run, "load_default_nperseg", return_value=[2048]):
            self.assertEqual(control_run.single_nperseg(None), 2048)

    def test_the_points_file_is_checked(self) -> None:
        points = load_points(self.points)
        self.assertEqual([point.name for point in points], ["tone", "mixed"])
        self.assertEqual(points[0].probes, (f"X:{TONE_HZ:.3f}",))
        with self.assertRaises(ValueError):
            select_points(points, ["nowhere"])
        doubled = Path(self.directory.name) / "doubled.toml"
        write_points(doubled, [("a", self.tone, []), ("a", self.mixed, [])])
        with self.assertRaises(ValueError):
            load_points(doubled)
        self.assertEqual(main(["../escape", "--points-file", str(self.points)]), 1)

    def test_the_repository_list_names_existing_folders(self) -> None:
        for point in load_points(control_run.POINTS_PATH):
            self.assertTrue(list(point.folder.glob("*_raw.npz")), point.name)


if __name__ == "__main__":
    unittest.main()
