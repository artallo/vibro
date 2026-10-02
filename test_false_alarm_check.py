"""Unit tests for the false alarm check of stable_spectrum.py.

Run with: python -m unittest test_false_alarm_check
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from false_alarm_check import (
    SAMPLES_PER_PACKET,
    binomial_interval,
    main,
    make_capture,
    measured_shape,
    run_check,
)

RATE_HZ = 250.0


def write_noise(path: Path, seed: int, packets: int = 64, line_hz: float = 0.0) -> None:
    generator = np.random.default_rng(seed)
    time = np.arange(packets * SAMPLES_PER_PACKET) / RATE_HZ
    axes = {}
    for key in "xyz":
        series = generator.normal(0.0, 1.0, packets * SAMPLES_PER_PACKET)
        if key == "z":
            # A low-frequency rise like the 1/f of the real Z axis.
            series += np.cumsum(generator.normal(0.0, 0.02, series.size))
        if line_hz and key == "y":
            series += 0.5 * np.sin(2.0 * np.pi * line_hz * time)
        axes[key] = series.reshape(packets, SAMPLES_PER_PACKET)
    np.savez(path, **axes, packet_fs_hz=np.full(packets, RATE_HZ))


class FalseAlarmTests(unittest.TestCase):
    def test_white_noise_gives_few_false_peaks(self) -> None:
        outcomes = run_check(
            runs=12, packets=64, windows_hz=[5.0, 10.0], nperseg_values=[1024],
            shape=None, rate_hz=RATE_HZ, seed=1,
        )
        self.assertEqual(len(outcomes), 2)
        for outcome in outcomes:
            self.assertEqual(outcome.runs, 12)
            self.assertLessEqual(outcome.runs_with_peaks, 2)

    def test_probes_on_white_noise_rarely_come_out_present(self) -> None:
        plain = run_check(
            runs=12, packets=64, windows_hz=[10.0], nperseg_values=[1024],
            shape=None, rate_hz=RATE_HZ, seed=7,
        )
        probed = run_check(
            runs=12, packets=64, windows_hz=[10.0], nperseg_values=[1024],
            shape=None, rate_hz=RATE_HZ, seed=7, probes=3,
        )
        # Probes draw from their own generator: the search is unchanged.
        self.assertEqual(plain[0].runs_with_peaks, probed[0].runs_with_peaks)
        self.assertLessEqual(probed[0].runs_with_probe_hits, 2)
        self.assertGreater(probed[0].probe_threshold, 0.0)

    def test_captures_are_continuous_and_coloured(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "noise_raw.npz"
            write_noise(path, 3)
            rate, shape = measured_shape([path])
        self.assertAlmostEqual(rate, RATE_HZ)
        from scipy.signal import welch

        axes = make_capture(np.random.default_rng(4), 32, shape)
        self.assertEqual(axes["z"].shape, (32, SAMPLES_PER_PACKET))
        # Z was given more low-frequency power than X, and keeps it.
        frequencies, z_psd = welch(axes["z"], RATE_HZ, nperseg=SAMPLES_PER_PACKET, axis=-1)
        _, x_psd = welch(axes["x"], RATE_HZ, nperseg=SAMPLES_PER_PACKET, axis=-1)
        low = (frequencies >= 0.25) & (frequencies <= 1.0)
        self.assertGreater(z_psd.mean(0)[low].mean(), 2.0 * x_psd.mean(0)[low].mean())

    def test_a_line_in_the_noise_captures_is_not_copied(self) -> None:
        line_hz = 40 * RATE_HZ / SAMPLES_PER_PACKET
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "noise_raw.npz"
            write_noise(path, 5, line_hz=line_hz)
            _, shape = measured_shape([path])
        neighbours = np.r_[shape["y"][34:38], shape["y"][43:47]].mean()
        self.assertLess(shape["y"][40] / neighbours, 1.3)

    def test_wilson_interval_brackets_the_rate(self) -> None:
        low, high = binomial_interval(3, 300)
        self.assertLess(low, 0.01)
        self.assertGreater(high, 0.01)
        self.assertEqual(binomial_interval(0, 0), (0.0, 0.0))

    def test_command_line_writes_a_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out"
            code = main([
                "--runs", "2", "--packets", "64", "--nperseg", "1024",
                "--baseline-window", "10", "--output", str(output),
            ])
            self.assertEqual(code, 0)
            report = (output / "false_alarm_report.txt").read_text(encoding="utf-8")
            self.assertIn("white Gaussian", report)
            self.assertIn("   10    1024", report)


if __name__ == "__main__":
    unittest.main()
