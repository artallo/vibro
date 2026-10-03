"""Unit tests for the control point runs.

Run with: python -m unittest test_control_run
"""

from __future__ import annotations

import argparse
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



class OldDetectorTests(unittest.TestCase):
    def test_only_the_welch_lines_change(self) -> None:
        from control_run import old_detector_config

        text = (
            "[serial]\r\nport = \"COM8\"\r\n\r\n[welch]\r\nnperseg = 1024\r\n"
            "noverlap = 512\r\n\r\n[stable_spectrum]\r\nnperseg = [2048]\r\n"
        )
        same, nperseg, noverlap = old_detector_config(text, None)
        self.assertEqual((same, nperseg, noverlap), (text, 1024, 512))
        changed, nperseg, noverlap = old_detector_config(text, 2048)
        self.assertEqual((nperseg, noverlap), (2048, 1024))
        self.assertEqual(
            changed,
            text.replace("nperseg = 1024", "nperseg = 2048").replace("noverlap = 512", "noverlap = 1024"),
        )
        self.assertIn("[stable_spectrum]\r\nnperseg = [2048]", changed)
        with self.assertRaises(ValueError):
            old_detector_config("[serial]\nport = 1\n", 2048)

    def test_layout_thresholds_go_into_their_own_table(self) -> None:
        from control_run import applied_thresholds, parse_threshold, set_layout_thresholds

        self.assertEqual(parse_threshold("2.0"), (None, 2.0))
        self.assertEqual(parse_threshold("8x16=2.95"), ("8x16", 2.95))
        for bad in ("8-16=2", "x=1", "8x16=a", "-1"):
            with self.assertRaises(argparse.ArgumentTypeError):
                parse_threshold(bad)
        text = (
            "[visualization.trusted_frequency]\r\nmin_median_prominence_db = 2.0\r\n\r\n"
            "[[analysis.bands]]\r\nprominence_db = 1.8\r\n"
        )
        added = set_layout_thresholds(text, {"8x16": 2.95, "8x8": 4.25})
        self.assertTrue(added.startswith(text))
        changed = set_layout_thresholds(added, {"8x16": 3.0})
        self.assertEqual(changed.count('"8x16"'), 1)
        self.assertIn('"8x16" = 3.0\r\n', changed)
        self.assertIn('"8x8" = 4.25\r\n', changed)
        self.assertEqual(
            applied_thresholds(changed, ["8x8", "8x16", "8x32"]),
            "8x8 4.25, 8x16 3, 8x32 2 (default)",
        )
        self.assertEqual(set_layout_thresholds(text, {}), text)

    def test_only_the_median_prominence_line_changes(self) -> None:
        from control_run import set_median_prominence

        text = (
            "[visualization.trusted_frequency]\r\nmin_support_fraction = 0.50\r\n"
            "min_median_prominence_db = 1.55\r\n\r\n[[analysis.bands]]\r\nprominence_db = 1.8\r\n"
        )
        self.assertEqual(set_median_prominence(text, None), (text, 1.55))
        changed, value = set_median_prominence(text, 2)
        self.assertEqual(value, 2)
        self.assertEqual(changed, text.replace("= 1.55", "= 2.0"))
        with self.assertRaises(ValueError):
            set_median_prominence("[welch]\nnperseg = 1024\n", 2.0)

    def test_regions_group_by_axis_and_frequency(self) -> None:
        from control_run import group_by_frequency

        items = [
            {"axis": "Y", "frequency_hz": 1.73}, {"axis": "X", "frequency_hz": 1.97},
            {"axis": "Y", "frequency_hz": 1.85}, {"axis": "Y", "frequency_hz": 2.30},
            {"axis": "X", "frequency_hz": 1.73},
        ]
        groups = group_by_frequency(items, 0.2)
        self.assertEqual(
            [[row["frequency_hz"] for row in group] for group in groups],
            [[1.73], [1.97], [1.73, 1.85], [2.30]],
        )

    def test_keys_of_one_detector_are_refused_with_the_other(self) -> None:
        for arguments in (
            ["v", "--layouts", "8x8"],
            ["v", "--median-prominence", "2.0"],
            ["v", "--old-detector", "--alpha", "0.05"],
            ["v", "--old-detector", "--separation-sigma", "3"],
        ):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                control_run.parse_cli_arguments(arguments)
        cli = control_run.parse_cli_arguments(["v", "--old-detector"])
        self.assertEqual(cli.layouts, control_run.DEFAULT_OLD_LAYOUTS)

    def test_replay_of_a_point_tables_its_trusted_regions(self) -> None:
        from false_alarm_check import save_capture

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "tone"
            folder.mkdir()
            for seed in range(2):
                generator = np.random.default_rng(seed)
                time = np.arange(64 * 1024) / RATE_HZ
                axes = {
                    key: generator.normal(0.0, 1.0, 64 * 1024) for key in "xyz"
                }
                axes["y"] += 0.4 * np.sin(2.0 * np.pi * 3.0 * time)
                axes = {key: value.reshape(64, 1024) for key, value in axes.items()}
                save_capture(folder / f"run{seed}_raw.npz", axes, RATE_HZ, "test")
            points = root / "points.toml"
            write_points(points, [("tone", folder, [])])
            output = root / "control"
            code = main([
                "old", "--old-detector", "--layouts", "8x8", "4x4", "--nperseg", "2048",
                "--median-prominence", "1.6", "4x4=99",
                "--points-file", str(points), "--output", str(output),
            ])
            self.assertEqual(code, 0)
            variant = output / "old"
            config = (variant / "config.toml").read_text(encoding="utf-8")
            self.assertIn("nperseg = 2048", config)
            self.assertIn("noverlap = 1024", config)
            self.assertIn("min_median_prominence_db = 1.6", config)
            self.assertTrue((variant / "tone" / "run0_raw" / "8x8" / "virtual_run01").is_dir())
            self.assertEqual(
                len(list((variant / "tone" / "run0_raw" / "4x4").glob("virtual_run*"))), 4,
            )
            rows = read_rows(variant / "control_old_peaks.csv")
            tone = [
                row for row in rows
                if row["axis"] == "Y" and abs(float(row["frequency_hz"]) - 3.0) < 0.2
            ]
            records = {row["record"] for row in tone}
            self.assertIn("run0_raw full", records)
            self.assertIn("run1_raw full", records)
            self.assertIn("across runs full", records)
            across = next(row for row in tone if row["record"] == "across runs full")
            self.assertEqual(across["runs"], "2/2")
            self.assertTrue(all(row["nperseg"] == "2048" for row in rows))
            summary = (variant / "control_summary.txt").read_text(encoding="utf-8")
            self.assertIn("Welch nperseg 2048, noverlap 1024", summary)
            self.assertIn("4x4 (4 runs)", summary)
            self.assertIn("runs with a trusted region: 8x8 2/2, 4x4 ", summary)
            self.assertIn("strongest trusted Med.Prom: 8x8 ", summary)
            # main.py takes the threshold of the layout from its own table.
            self.assertIn("Med.Prom thresholds, dB: 8x8 1.6 (default), 4x4 99", summary)
            self.assertIn("runs with a trusted region: 8x8 2/2, 4x4 0/8", summary)
            log = variant / "tone" / "run0_raw" / "4x4" / "virtual_run01" / "result.txt"
            self.assertIn("Trusted Med.Prom threshold: 99 dB (layout 4x4)", log.read_text(encoding="utf-8"))
            log = variant / "tone" / "run0_raw" / "8x8" / "virtual_run01" / "result.txt"
            self.assertIn("Trusted Med.Prom threshold: 1.6 dB (default)", log.read_text(encoding="utf-8"))
            # Runs of the two detectors are never compared with each other.
            stable = output / "stable"
            stable.mkdir()
            (stable / "control_peaks.csv").write_text(
                ",".join(control_run.PEAK_FIELDS) + "\n", encoding="utf-8",
            )
            self.assertEqual(main(["--compare", "old", "stable", "--output", str(output)]), 1)


if __name__ == "__main__":
    unittest.main()
