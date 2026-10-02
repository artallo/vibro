"""How often stable_spectrum.py finds a peak in pure noise.

The threshold of ``stable_spectrum.py`` promises at most ``alpha`` captures
with a false peak (1 % by default). This script checks that promise for a
given baseline window and set of segment lengths: it makes many captures of
pure noise, runs the same peak search on each and counts the captures that
come out with at least one peak.

Two kinds of noise:

* white Gaussian noise on every axis (default);
* Gaussian noise coloured like the real sensor noise (``--noise-shape``):
  the mean spectrum of captures that hold only noise, axis by axis and
  smoothed by a running median of about 2 Hz, is imposed on a continuous
  random series. This keeps the 1/f rise of Z at the low end of the band,
  where a wide baseline is most at risk, but no weak line those captures
  may hold: every realisation is new noise.

Captures are made as one continuous series and cut into packets, so long
segments that span packet joins see no artificial steps.

With ``--probes K`` every capture is also probed at K frequencies drawn at
random in the band, each on one random axis, as ``stable_spectrum.py
--probe`` and ``--confirm-from`` do: the threshold counts only the bins in
the probe windows. A capture counts when any probe comes out "present".
This checks the honest threshold for frequencies chosen in advance; the
frequencies come from their own random generator, so the search results
do not change when probes are added.

Usage::

    python false_alarm_check.py --runs 300
    python false_alarm_check.py --runs 300 --probes 3
    python false_alarm_check.py --runs 300 --baseline-window 5 10 \\
        --noise-shape "tumen_results/260912 Тюмень/*_raw.npz" "tumen_results/260916 Тюмень, три прогона/*_raw.npz"
"""

from __future__ import annotations

import argparse
import glob
import sys
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np

from stable_spectrum import (
    AXIS_KEYS,
    DEFAULT_NPERSEG_SWEEP,
    STABLE_RESULTS_DIRECTORY,
    DEFAULT_PROBE_TOLERANCE_HZ,
    ProbeRequest,
    analyze_axis,
    drop_startup_packet,
    find_stable_peaks,
    load_band_names,
    load_settings,
    probe_frequencies,
    probe_threshold,
    rolling_median,
    significance_threshold,
)

SAMPLES_PER_PACKET = 1024
DEFAULT_RATE_HZ = 252.7
# The noise shape keeps only what is broader than this, so a weak real line
# in the noise captures does not end up inside every generated capture.
SHAPE_SMOOTHING_HZ = 2.0


@dataclass(frozen=True)
class Outcome:
    window_hz: float
    nperseg: int
    runs: int
    runs_with_peaks: int
    peaks: list[tuple[str, float, float]]  # axis, frequency, z
    runs_with_probe_hits: int = 0
    probe_hits: list[tuple[str, float, float]] = ()  # axis, frequency, z
    probe_threshold: float = 0.0

    @property
    def rate(self) -> float:
        return self.runs_with_peaks / self.runs if self.runs else 0.0

    @property
    def probe_rate(self) -> float:
        return self.runs_with_probe_hits / self.runs if self.runs else 0.0


def expand(items: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in items:
        candidate = Path(item)
        if candidate.is_dir():
            paths.extend(sorted(candidate.glob("*.npz")))
        elif any(character in item for character in "*?["):
            paths.extend(Path(match) for match in sorted(glob.glob(item)))
        else:
            paths.append(candidate)
    if not paths:
        raise ValueError(f"no captures match {' '.join(items)}")
    return paths


def measured_shape(paths: list[Path]) -> tuple[float, dict[str, np.ndarray]]:
    """Mean amplitude spectrum per axis over every packet, and the mean rate."""
    sums = {key: np.zeros(SAMPLES_PER_PACKET // 2 + 1) for _, key in AXIS_KEYS}
    count = 0
    rates = []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            axes, packet_fs_hz, _ = drop_startup_packet(
                {key: np.asarray(archive[key], dtype=float) for _, key in AXIS_KEYS},
                np.asarray(archive["packet_fs_hz"], dtype=float),
            )
        if axes["x"].shape[1] != SAMPLES_PER_PACKET:
            raise ValueError(f"{path.name}: packets of {axes['x'].shape[1]} samples")
        rates.append(float(np.mean(packet_fs_hz)))
        window = np.hanning(SAMPLES_PER_PACKET)
        for _, key in AXIS_KEYS:
            packets = axes[key] - axes[key].mean(axis=1, keepdims=True)
            sums[key] += np.sum(np.abs(np.fft.rfft(packets * window, axis=1)) ** 2, axis=0)
        count += axes["x"].shape[0]
    rate_hz = float(np.mean(rates))
    smoothing_bins = max(3, int(round(SHAPE_SMOOTHING_HZ * SAMPLES_PER_PACKET / rate_hz)))
    shape = {}
    for _, key in AXIS_KEYS:
        amplitude = np.sqrt(sums[key] / count)
        amplitude[0] = amplitude[1]
        amplitude = rolling_median(amplitude, smoothing_bins)
        shape[key] = amplitude / np.sqrt(np.mean(amplitude ** 2))
    return rate_hz, shape


def make_capture(
    generator: np.random.Generator,
    packets: int,
    shape: dict[str, np.ndarray] | None,
) -> dict[str, np.ndarray]:
    """One capture of pure noise, continuous across packet joins."""
    samples = packets * SAMPLES_PER_PACKET
    axes = {}
    for _, key in AXIS_KEYS:
        series = generator.normal(0.0, 1.0, samples)
        if shape is not None:
            spectrum = np.fft.rfft(series)
            grid = np.linspace(0.0, 1.0, spectrum.size)
            reference = np.linspace(0.0, 1.0, shape[key].size)
            spectrum *= np.interp(grid, reference, shape[key])
            series = np.fft.irfft(spectrum, n=samples)
        axes[key] = series.reshape(packets, SAMPLES_PER_PACKET)
    return axes


def peaks_in(
    axes: dict[str, np.ndarray],
    rate_hz: float,
    settings,
    bands,
    spectra=None,
) -> list:
    if spectra is None:
        spectra = [analyze_axis(axis, axes[key], rate_hz, settings) for axis, key in AXIS_KEYS]
    bins_tested = sum(spectrum.frequencies.size for spectrum in spectra)
    threshold = significance_threshold(
        bins_tested, settings.alpha, int(round(spectra[0].effective_rows)),
    )
    return find_stable_peaks(spectra, threshold, settings, bands)


def random_probes(
    generator: np.random.Generator,
    count: int,
    band_hz: tuple[float, float],
    tolerance_hz: float,
) -> list[ProbeRequest]:
    """Frequencies picked without looking at the capture, one axis each."""
    low, high = band_hz[0] + tolerance_hz, band_hz[1] - tolerance_hz
    axes = [name for name, _ in AXIS_KEYS]
    return [
        ProbeRequest(float(generator.uniform(low, high)), str(generator.choice(axes)))
        for _ in range(count)
    ]


def probe_hits_in(spectra, requests, tolerance_hz: float, alpha: float):
    rows = int(round(spectra[0].effective_rows))
    threshold, _ = probe_threshold(spectra, requests, tolerance_hz, alpha, rows)
    results = probe_frequencies(spectra, requests, threshold, tolerance_hz)
    return [probe for probe in results if probe.detected], threshold


def run_check(
    runs: int,
    packets: int,
    windows_hz: list[float],
    nperseg_values: list[int],
    shape: dict[str, np.ndarray] | None,
    rate_hz: float,
    seed: int,
    band_hz: tuple[float, float] | None = None,
    alpha: float = 0.01,
    progress: bool = False,
    probes: int = 0,
    probe_tolerance_hz: float = DEFAULT_PROBE_TOLERANCE_HZ,
    probe_alpha: float = 0.01,
) -> list[Outcome]:
    """Every window and segment length sees the same noise captures."""
    bands = load_band_names()
    base = load_settings(alpha, band_hz)
    probe_generator = np.random.default_rng([seed, 1])
    probe_hits = {}
    probe_found = {}
    probe_thresholds = {}
    settings = {
        (window, nperseg): replace(
            base, nperseg=nperseg, noverlap=nperseg // 2, baseline_window_hz=window,
        )
        for window in windows_hz
        for nperseg in nperseg_values
    }
    hits = {key: 0 for key in settings}
    found = {key: [] for key in settings}
    generator = np.random.default_rng(seed)
    for run in range(runs):
        axes = make_capture(generator, packets, shape)
        requests = random_probes(probe_generator, probes, base.band_hz, probe_tolerance_hz)
        for key, current in settings.items():
            spectra = [
                analyze_axis(axis, axes[axis_key], rate_hz, current)
                for axis, axis_key in AXIS_KEYS
            ]
            peaks = peaks_in(axes, rate_hz, current, bands, spectra)
            if peaks:
                hits[key] += 1
                found[key].extend((peak.axis, peak.frequency_hz, peak.z) for peak in peaks)
            if requests:
                detected, threshold = probe_hits_in(
                    spectra, requests, probe_tolerance_hz, probe_alpha,
                )
                probe_thresholds.setdefault(key, []).append(threshold)
                if detected:
                    probe_hits[key] = probe_hits.get(key, 0) + 1
                    probe_found.setdefault(key, []).extend(
                        (probe.axis, probe.frequency_hz, probe.z) for probe in detected
                    )
        if progress and (run + 1) % 25 == 0:
            print(f"  {run + 1}/{runs} captures", flush=True)
    return [
        Outcome(
            window, nperseg, runs, hits[(window, nperseg)], found[(window, nperseg)],
            probe_hits.get((window, nperseg), 0),
            probe_found.get((window, nperseg), []),
            float(np.median(probe_thresholds[(window, nperseg)]))
            if (window, nperseg) in probe_thresholds else 0.0,
        )
        for window, nperseg in settings
    ]


def binomial_interval(hits: int, runs: int) -> tuple[float, float]:
    """95 % Wilson interval for a rate."""
    if runs == 0:
        return 0.0, 0.0
    z = 1.96
    rate = hits / runs
    centre = (rate + z * z / (2 * runs)) / (1 + z * z / runs)
    half = z * np.sqrt(rate * (1 - rate) / runs + z * z / (4 * runs * runs)) / (1 + z * z / runs)
    return max(0.0, centre - half), min(1.0, centre + half)


def format_report(
    outcomes: list[Outcome],
    noise_kind: str,
    packets: int,
    rate_hz: float,
    alpha: float,
    seed: int,
    probes: int = 0,
    probe_tolerance_hz: float = DEFAULT_PROBE_TOLERANCE_HZ,
    probe_alpha: float = 0.01,
) -> str:
    lines = [
        "False alarm check of stable_spectrum.py",
        f"Created: {datetime.now().isoformat(timespec='seconds')}",
        f"Noise: {noise_kind}",
        f"Captures of {packets} packets x {SAMPLES_PER_PACKET} samples at {rate_hz:.2f} Hz, "
        f"seed {seed}, alpha {alpha:g}",
        "",
        "A capture counts when the peak search finds at least one peak in it.",
        f"Expected rate: about alpha = {alpha:.1%}, at most that by design.",
        "",
        f"{'Window Hz':>9s} {'nperseg':>7s} {'runs':>5s} {'with peak':>9s} "
        f"{'rate':>6s} {'95% interval':>15s}  peaks (axis Hz z)",
    ]
    for outcome in outcomes:
        low, high = binomial_interval(outcome.runs_with_peaks, outcome.runs)
        listed = ", ".join(
            f"{axis} {frequency:.2f} z{z:.1f}" for axis, frequency, z in outcome.peaks[:12]
        )
        if len(outcome.peaks) > 12:
            listed += f", ... {len(outcome.peaks) - 12} more"
        lines.append(
            f"{outcome.window_hz:9g} {outcome.nperseg:7d} {outcome.runs:5d} "
            f"{outcome.runs_with_peaks:9d} {outcome.rate:6.1%} "
            f"{low:6.1%}-{high:6.1%}  {listed}"
        )
    if probes:
        lines += [
            "",
            f"Probes: {probes} random frequencies per capture, one axis each, "
            f"window +-{probe_tolerance_hz:g} Hz, alpha {probe_alpha:g}.",
            "A capture counts when any probe comes out present. Expected rate:",
            f"about alpha = {probe_alpha:.1%}.",
            "",
            f"{'Window Hz':>9s} {'nperseg':>7s} {'runs':>5s} {'present':>9s} "
            f"{'rate':>6s} {'95% interval':>15s} {'thr':>5s}  hits (axis Hz z)",
        ]
        for outcome in outcomes:
            low, high = binomial_interval(outcome.runs_with_probe_hits, outcome.runs)
            listed = ", ".join(
                f"{axis} {frequency:.2f} z{z:.1f}"
                for axis, frequency, z in list(outcome.probe_hits)[:12]
            )
            lines.append(
                f"{outcome.window_hz:9g} {outcome.nperseg:7d} {outcome.runs:5d} "
                f"{outcome.runs_with_probe_hits:9d} {outcome.probe_rate:6.1%} "
                f"{low:6.1%}-{high:6.1%} {outcome.probe_threshold:5.2f}  {listed}"
            )
    return "\n".join(lines) + "\n"


def parse_cli_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count false peaks of stable_spectrum.py on pure noise",
    )
    parser.add_argument("--runs", type=int, default=300, help="noise captures (default 300)")
    parser.add_argument(
        "--packets", type=int, default=256, help="packets per capture (default 256)",
    )
    parser.add_argument(
        "--baseline-window", type=float, nargs="+", default=None, metavar="HZ",
        help="baseline windows to check (default: from config.toml)",
    )
    parser.add_argument(
        "--nperseg", type=int, nargs="+", default=list(DEFAULT_NPERSEG_SWEEP),
        metavar="SAMPLES", help="segment lengths (default 1024 2048 4096)",
    )
    parser.add_argument(
        "--noise-shape", nargs="+", default=None, metavar="PATH",
        help="captures holding only noise; their mean spectrum colours the "
             "generated noise (default: white noise)",
    )
    parser.add_argument(
        "--band", type=float, nargs=2, default=None, metavar=("MIN_HZ", "MAX_HZ"),
        help="analysis band (default: from config.toml)",
    )
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument(
        "--probes", type=int, default=0, metavar="K",
        help="also probe K random frequencies per capture with the honest "
             "threshold of --probe and --confirm-from (default 0: off)",
    )
    parser.add_argument(
        "--probe-tolerance", type=float, default=DEFAULT_PROBE_TOLERANCE_HZ, metavar="HZ",
    )
    parser.add_argument("--probe-alpha", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output", type=Path, default=None,
        help="folder for the report (default: stable_results/false_alarm_<time>/)",
    )
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    cli = parse_cli_arguments(arguments)
    band = tuple(cli.band) if cli.band else None
    windows = cli.baseline_window or [load_settings(cli.alpha, band).baseline_window_hz]
    if cli.noise_shape:
        try:
            rate_hz, shape = measured_shape(expand(cli.noise_shape))
        except (OSError, ValueError, KeyError) as error:
            print(f"error: {error}")
            return 1
        noise_kind = "coloured like " + ", ".join(cli.noise_shape)
    else:
        rate_hz, shape, noise_kind = DEFAULT_RATE_HZ, None, "white Gaussian"
    outcomes = run_check(
        cli.runs, cli.packets, windows, cli.nperseg, shape, rate_hz, cli.seed,
        band, cli.alpha, progress=True, probes=cli.probes,
        probe_tolerance_hz=cli.probe_tolerance, probe_alpha=cli.probe_alpha,
    )
    report = format_report(
        outcomes, noise_kind, cli.packets, rate_hz, cli.alpha, cli.seed,
        cli.probes, cli.probe_tolerance, cli.probe_alpha,
    )
    output = cli.output or (
        STABLE_RESULTS_DIRECTORY / f"false_alarm_{datetime.now():%Y%m%d_%H%M%S}"
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "false_alarm_report.txt").write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"Saved: {output / 'false_alarm_report.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
