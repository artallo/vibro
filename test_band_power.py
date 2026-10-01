"""Unit tests for band power against the sensor noise floor.

Run with: python -m unittest test_band_power
"""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

from band_power import (
    analyse,
    load_point,
    main,
    merge_structures,
    parse_band,
    scan_bands,
)

RATE_HZ = 250.0
PACKET = 1024
HUMP_HZ = (9.0, 12.0)


def write_capture(
    path: Path,
    seed: int,
    packets: int = 96,
    rate_hz: float = RATE_HZ,
    hump: float = 0.0,
) -> None:
    """White noise on every axis, plus band-limited noise on X when hump > 0."""
    generator = np.random.default_rng(seed)
    axes = {key: generator.normal(0.0, 1.0, packets * PACKET) for key in "xyz"}
    if hump:
        sos = butter(4, HUMP_HZ, btype="bandpass", fs=rate_hz, output="sos")
        axes["x"] += hump * sosfiltfilt(sos, generator.normal(0.0, 1.0, packets * PACKET))
    np.savez(
        path,
        **{key: values.reshape(packets, PACKET) for key, values in axes.items()},
        packet_fs_hz=np.full(packets, rate_hz),
    )


class BandPowerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.folder = Path(self.directory.name)
        for seed in range(3):
            write_capture(self.folder / f"noise{seed}_raw.npz", 100 + seed)
        for seed in range(2):
            write_capture(self.folder / f"floor{seed}_raw.npz", 200 + seed, hump=0.7)
            write_capture(self.folder / f"quiet{seed}_raw.npz", 300 + seed)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def points(self):
        noise = load_point("noise", [str(self.folder / "noise*_raw.npz")], PACKET)
        floor = load_point("floor", [str(self.folder / "floor*_raw.npz")], PACKET)
        quiet = load_point("quiet", [str(self.folder / "quiet*_raw.npz")], PACKET)
        return noise, floor, quiet

    def test_broad_hump_is_found_on_its_axis_only(self) -> None:
        noise, floor, _ = self.points()
        results = analyse([floor], noise, [HUMP_HZ], alpha=0.01)
        by_axis = {result.axis: result for result in results}
        self.assertTrue(by_axis["X"].significant)
        self.assertGreater(by_axis["X"].excess_db, 0.5)
        self.assertFalse(by_axis["Y"].significant)
        self.assertFalse(by_axis["Z"].significant)

    def test_scan_places_the_hump_and_noise_gives_nothing(self) -> None:
        noise, floor, quiet = self.points()
        scan = scan_bands((0.5, 30.0), 2.0)
        structures = merge_structures(analyse([floor, quiet], noise, scan, alpha=0.01))
        self.assertTrue(structures)
        self.assertEqual({structure.point for structure in structures}, {"floor"})
        self.assertEqual({structure.axis for structure in structures}, {"X"})
        for structure in structures:
            self.assertLess(structure.low_hz, HUMP_HZ[1])
            self.assertGreater(structure.high_hz, HUMP_HZ[0])

    def test_narrow_baseline_hides_the_hump_that_band_power_sees(self) -> None:
        noise, floor, _ = self.points()
        result = next(
            item for item in analyse([floor], noise, [HUMP_HZ], 0.01, 0.0, [5.0, 15.0])
            if item.axis == "X"
        )
        self.assertLess(result.baseline_prominence_db[5.0], result.excess_db)
        self.assertLess(
            result.baseline_prominence_db[5.0], result.baseline_prominence_db[15.0],
        )

    def test_systematic_term_lowers_z(self) -> None:
        noise, floor, _ = self.points()
        plain = analyse([floor], noise, [HUMP_HZ], 0.01)[0]
        careful = analyse([floor], noise, [HUMP_HZ], 0.01, systematic_db=0.3)[0]
        self.assertLess(careful.z, plain.z)
        self.assertAlmostEqual(careful.excess_db, plain.excess_db)

    def test_start_up_packet_is_left_out(self) -> None:
        path = self.folder / "startup_raw.npz"
        write_capture(path, 400, packets=8)
        with np.load(path) as archive:
            data = {key: archive[key].copy() for key in archive.files}
        data["z"][0, :] += np.linspace(0.0, 1.0e4, PACKET)
        np.savez(path, **data)
        point = load_point("startup", [str(path)], PACKET)
        self.assertEqual(point.packet_count, 7)

    def test_mixed_rates_are_refused(self) -> None:
        write_capture(self.folder / "slow_raw.npz", 500, rate_hz=125.0)
        code = main([
            "--point", "slow", str(self.folder / "slow_raw.npz"),
            "--noise", str(self.folder / "noise0_raw.npz"),
            "--output", str(self.folder / "out_rates"),
        ])
        self.assertEqual(code, 1)

    def test_command_line_writes_report_csv_and_figure(self) -> None:
        output = self.folder / "out"
        code = main([
            "--point", "floor", str(self.folder / "floor*_raw.npz"),
            "--point", "quiet", str(self.folder / "quiet0_raw.npz"),
            str(self.folder / "quiet1_raw.npz"),
            "--noise", str(self.folder / "noise*_raw.npz"),
            "--bands", "9-12", "--scan", "2", "--band", "0.5", "30",
            "--output", str(output),
        ])
        self.assertEqual(code, 0)
        for name in ("band_power_report.txt", "band_power.csv", "figure_band_power.png"):
            self.assertTrue((output / name).exists(), name)
        report = (output / "band_power_report.txt").read_text(encoding="utf-8")
        self.assertIn("above reference", report)

    def test_reference_point_mode(self) -> None:
        output = self.folder / "out_ref"
        code = main([
            "--point", "floor", str(self.folder / "floor*_raw.npz"),
            "--point", "quiet", str(self.folder / "quiet*_raw.npz"),
            "--reference", "quiet", "--bands", "9-12",
            "--output", str(output),
        ])
        self.assertEqual(code, 0)
        report = (output / "band_power_report.txt").read_text(encoding="utf-8")
        self.assertIn("Reference (measuring point): quiet", report)

    def test_folder_is_accepted_as_a_point(self) -> None:
        sub = self.folder / "sub"
        sub.mkdir()
        write_capture(sub / "a_raw.npz", 600)
        point = load_point("sub", [str(sub)], PACKET)
        self.assertEqual(len(point.captures), 1)


class ParsingTests(unittest.TestCase):
    def test_band_text(self) -> None:
        self.assertEqual(parse_band("9.5-13"), (9.5, 13.0))
        self.assertEqual(parse_band("9,5-13"), (9.5, 13.0))
        for wrong in ("13-9.5", "abc", "9.5"):
            with self.assertRaises(argparse.ArgumentTypeError):
                parse_band(wrong)

    def test_scan_windows_step_by_half(self) -> None:
        windows = scan_bands((0.0, 4.0), 2.0)
        self.assertEqual(windows, [(0.0, 2.0), (1.0, 3.0), (2.0, 4.0)])
        with self.assertRaises(ValueError):
            scan_bands((0.0, 1.0), 2.0)


if __name__ == "__main__":
    unittest.main()
