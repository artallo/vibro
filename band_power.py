"""Power in frequency bands against the sensor noise floor.

Motivation
----------
``stable_spectrum.py`` looks for peaks: each bin is compared with a
running median of its neighbours (10 Hz wide, 5 Hz until 2026-10-01). A
structure wider than about half that window becomes its own baseline, and
a weak hump that sits under the sensor noise rises only a fraction of a dB
above it. This script asks the complementary question:

    how much more power is there in a band than the sensor itself makes?

Method
------
1. Each capture is cut into one periodogram per packet (Hann, ``nperseg``
   up to the packet length), with the start-up packet left out as in
   ``stable_spectrum.py``. Captures given for one measuring point are
   pooled packet by packet.
2. The power in a band is the sum of the periodogram over the band, packet
   by packet. Its mean and standard error come from the spread across
   packets, so an intermittent signal widens its own error bar.
3. The reference is the same quantity from captures that hold only sensor
   noise (``--noise``), or from another measuring point (``--reference``),
   for example the basement under the floor of interest.
4. ``z = (P - P_ref) / sqrt(se^2 + se_ref^2 + sys^2)``, where ``sys`` is an
   optional systematic uncertainty of the reference in dB
   (``--noise-sys-db``). The report lists how much the noise captures
   themselves differ from each other in every band, which is the honest
   size of that systematic term.
5. Bands are either named (``--bands 9.5-13``) or scanned: windows of
   ``--scan`` Hz with a half-window step across ``--band``. The threshold
   is the Bonferroni-corrected Student quantile over all bands and axes of
   a point, as in ``stable_spectrum.py``. Neighbouring scan windows overlap
   by half, so the correction is conservative. Adjacent significant
   windows are reported as one structure.

The figure shows the amplitude spectrum of every point and of the
reference, plus the running-median baselines of the first point for the
windows of ``--baseline-windows``, so it is visible how a baseline follows a
broad hump. The report also gives, for named bands, the mean prominence
over each such baseline: what ``stable_spectrum.py`` sees of the band.

No building frequency is built in: bands come from the command line or
from the scan.

Usage::

    python band_power.py --point "floor 5" "path/to/folder" \\
        --noise "tumen_results/20260916_*_raw.npz"
    python band_power.py --point "stair 1, 5" a_raw.npz b_raw.npz \\
        --point "basement" c_raw.npz d_raw.npz \\
        --noise "tumen_results/20260912_*_raw.npz" "tumen_results/20260916_*_raw.npz" \\
        --bands 9.5-13 9-12 --band 0.5 30
    python band_power.py --point "floor 5" a_raw.npz --point "basement" c_raw.npz \\
        --reference "basement"
"""

from __future__ import annotations

import argparse
import csv
import glob
import sys
import zipfile
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

import numpy as np

from stable_spectrum import (
    AXIS_KEYS,
    STABLE_RESULTS_DIRECTORY,
    drop_startup_packet,
    load_settings,
    packet_periodograms,
    rolling_median,
    significance_threshold,
)

DEFAULT_SCAN_WIDTH_HZ = 2.0
DEFAULT_PLOT_BAND_HZ = (0.5, 30.0)
# Captures whose mean sampling rates differ by more than this are not
# compared: the anti-alias filter and the frequency grid would differ.
MAX_RATE_MISMATCH = 0.005
G_TO_UG = 1.0e6
# Categorical slots in fixed order, one per measuring point.
POINT_COLORS = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
)
REFERENCE_COLOR = "#8a8a85"
BAND_SHADE_COLOR = "#eda100"


@dataclass(frozen=True)
class CaptureSpectra:
    """Per-packet periodograms of one capture, one array per axis."""

    name: str
    sampling_rate_hz: float
    frequencies: np.ndarray
    psd: dict[str, np.ndarray]  # axis -> (packets, bins), g^2/Hz


@dataclass(frozen=True)
class Point:
    label: str
    captures: list[CaptureSpectra]

    @property
    def packet_count(self) -> int:
        return sum(capture.psd["X"].shape[0] for capture in self.captures)

    @property
    def sampling_rate_hz(self) -> float:
        return float(np.mean([capture.sampling_rate_hz for capture in self.captures]))

    def band_power(self, axis: str, low_hz: float, high_hz: float) -> np.ndarray:
        """Power in [low, high] for every packet of every capture, g^2."""
        values = []
        for capture in self.captures:
            selected = (capture.frequencies >= low_hz) & (capture.frequencies <= high_hz)
            bin_width = float(capture.frequencies[1] - capture.frequencies[0])
            values.append(capture.psd[axis][:, selected].sum(axis=1) * bin_width)
        return np.concatenate(values)

    def mean_psd(self, axis: str) -> tuple[np.ndarray, np.ndarray]:
        """Mean PSD over all packets, on the grid of the first capture."""
        grid = self.captures[0].frequencies
        weighted = np.zeros_like(grid)
        for capture in self.captures:
            mean = capture.psd[axis].mean(axis=0)
            weighted += np.interp(grid, capture.frequencies, mean) * capture.psd[axis].shape[0]
        return grid, weighted / self.packet_count


@dataclass(frozen=True)
class BandResult:
    point: str
    axis: str
    low_hz: float
    high_hz: float
    power: float
    power_se: float
    reference_power: float
    reference_se: float
    excess_db: float
    z: float
    signal_density_ug: float
    threshold: float
    reference_spread_db: float = float("nan")
    baseline_prominence_db: dict[float, float] = field(default_factory=dict)

    @property
    def significant(self) -> bool:
        return self.z >= self.threshold


@dataclass(frozen=True)
class Structure:
    """Adjacent significant scan windows of one point and axis."""

    point: str
    axis: str
    low_hz: float
    high_hz: float
    strongest: BandResult


# ==========================================================
# Loading
# ==========================================================

def expand_paths(items: list[str]) -> list[Path]:
    """Files, glob patterns (expanded here, so PowerShell works) and folders."""
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


def load_capture(path: Path, nperseg: int) -> CaptureSpectra | None:
    try:
        archive = np.load(path, allow_pickle=False)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(f"skipped {path.name}: not readable ({error})")
        return None
    with archive:
        missing = [key for key in ("x", "y", "z", "packet_fs_hz") if key not in archive.files]
        if missing:
            print(f"skipped {path.name}: no {', '.join(missing)}")
            return None
        axes, packet_fs_hz, _ = drop_startup_packet(
            {key: np.asarray(archive[key], dtype=float) for _, key in AXIS_KEYS},
            np.asarray(archive["packet_fs_hz"], dtype=float),
        )
    sampling_rate_hz = float(np.mean(packet_fs_hz))
    segment = min(nperseg, axes["x"].shape[1])
    settings = replace(load_settings(0.01, None), nperseg=segment, noverlap=segment // 2)
    psd = {}
    frequencies = None
    for axis, key in AXIS_KEYS:
        frequencies, psd[axis] = packet_periodograms(axes[key], sampling_rate_hz, settings)
    return CaptureSpectra(
        name=path.stem,
        sampling_rate_hz=sampling_rate_hz,
        frequencies=frequencies,
        psd=psd,
    )


def load_point(label: str, items: list[str], nperseg: int) -> Point:
    captures = [
        capture
        for capture in (load_capture(path, nperseg) for path in expand_paths(items))
        if capture is not None
    ]
    if not captures:
        raise ValueError(f"point '{label}': no readable captures")
    return Point(label=label, captures=captures)


def check_rates(points: list[Point]) -> None:
    rates = {
        f"{point.label} / {capture.name}": capture.sampling_rate_hz
        for point in points
        for capture in point.captures
    }
    values = np.array(list(rates.values()))
    if np.ptp(values) > MAX_RATE_MISMATCH * float(np.mean(values)):
        listed = ", ".join(f"{name} {rate:.2f} Hz" for name, rate in rates.items())
        raise ValueError(
            f"captures were recorded at different sampling rates ({listed}); "
            "compare only one ODR at a time"
        )


# ==========================================================
# Statistics
# ==========================================================

def parse_band(text: str) -> tuple[float, float]:
    try:
        low_text, high_text = text.replace(",", ".").split("-")
        low, high = float(low_text), float(high_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"band '{text}' is not of the form LOW-HIGH, e.g. 9.5-13"
        ) from error
    if not 0.0 <= low < high:
        raise argparse.ArgumentTypeError(f"band '{text}' must have 0 <= LOW < HIGH")
    return low, high


def scan_bands(band_hz: tuple[float, float], width_hz: float) -> list[tuple[float, float]]:
    low, high = band_hz
    if width_hz <= 0.0 or width_hz > high - low:
        raise ValueError(f"scan width {width_hz:g} Hz does not fit into {low:g}-{high:g} Hz")
    step = width_hz / 2.0
    starts = np.arange(low, high - width_hz + 1.0e-9, step)
    return [(float(start), float(start + width_hz)) for start in starts]


def mean_and_se(values: np.ndarray) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=1) / np.sqrt(values.size))


def reference_spread_db(reference: Point, axis: str, low_hz: float, high_hz: float) -> float:
    """How much the reference captures differ from each other in this band."""
    if len(reference.captures) < 2:
        return float("nan")
    pooled = float(np.mean(reference.band_power(axis, low_hz, high_hz)))
    per_capture = [
        float(np.mean(
            Point(reference.label, [capture]).band_power(axis, low_hz, high_hz)
        ))
        for capture in reference.captures
    ]
    return float(np.std(10.0 * np.log10(np.array(per_capture) / pooled), ddof=1))


def baseline_prominence(
    point: Point, axis: str, low_hz: float, high_hz: float, windows_hz: list[float],
) -> dict[float, float]:
    """Mean prominence over running-median baselines inside the band, dB."""
    frequencies, mean = point.mean_psd(axis)
    bin_width = float(frequencies[1] - frequencies[0])
    usable = frequencies > 0.0
    result = {}
    for window_hz in windows_hz:
        baseline = np.full_like(mean, np.nan)
        baseline[usable] = rolling_median(
            mean[usable], max(3, int(round(window_hz / bin_width))),
        )
        selected = usable & (frequencies >= low_hz) & (frequencies <= high_hz)
        result[window_hz] = float(np.mean(10.0 * np.log10(mean[selected] / baseline[selected])))
    return result


def compare_band(
    point: Point,
    reference: Point,
    axis: str,
    low_hz: float,
    high_hz: float,
    threshold: float,
    systematic_db: float = 0.0,
    baseline_windows_hz: list[float] | None = None,
) -> BandResult:
    power, power_se = mean_and_se(point.band_power(axis, low_hz, high_hz))
    reference_power, reference_se = mean_and_se(reference.band_power(axis, low_hz, high_hz))
    systematic = reference_power * (10.0 ** (systematic_db / 10.0) - 1.0)
    error = float(np.sqrt(power_se ** 2 + reference_se ** 2 + systematic ** 2))
    excess = power - reference_power
    return BandResult(
        point=point.label,
        axis=axis,
        low_hz=low_hz,
        high_hz=high_hz,
        power=power,
        power_se=power_se,
        reference_power=reference_power,
        reference_se=reference_se,
        excess_db=float(10.0 * np.log10(power / reference_power)),
        z=float(excess / error) if error > 0.0 else 0.0,
        signal_density_ug=float(np.sqrt(max(excess, 0.0) / (high_hz - low_hz)) * G_TO_UG),
        threshold=threshold,
        reference_spread_db=reference_spread_db(reference, axis, low_hz, high_hz),
        baseline_prominence_db=(
            baseline_prominence(point, axis, low_hz, high_hz, baseline_windows_hz)
            if baseline_windows_hz else {}
        ),
    )


def analyse(
    points: list[Point],
    reference: Point,
    bands: list[tuple[float, float]],
    alpha: float,
    systematic_db: float = 0.0,
    baseline_windows_hz: list[float] | None = None,
) -> list[BandResult]:
    """Every band, axis and point; the threshold covers bands x axes of a point."""
    results = []
    for point in points:
        threshold = significance_threshold(
            len(bands) * len(AXIS_KEYS), alpha, point.packet_count,
        )
        for axis, _ in AXIS_KEYS:
            for low_hz, high_hz in bands:
                results.append(compare_band(
                    point, reference, axis, low_hz, high_hz, threshold,
                    systematic_db, baseline_windows_hz,
                ))
    return results


def merge_structures(results: list[BandResult]) -> list[Structure]:
    """Join overlapping or touching significant scan windows per point and axis."""
    structures: list[Structure] = []
    keyed: dict[tuple[str, str], list[BandResult]] = {}
    for result in results:
        if result.significant:
            keyed.setdefault((result.point, result.axis), []).append(result)
    for (point, axis), hits in keyed.items():
        hits.sort(key=lambda item: item.low_hz)
        group = [hits[0]]
        for hit in hits[1:]:
            if hit.low_hz <= max(item.high_hz for item in group) + 1.0e-9:
                group.append(hit)
                continue
            structures.append(_structure(point, axis, group))
            group = [hit]
        structures.append(_structure(point, axis, group))
    return structures


def _structure(point: str, axis: str, group: list[BandResult]) -> Structure:
    return Structure(
        point=point,
        axis=axis,
        low_hz=min(item.low_hz for item in group),
        high_hz=max(item.high_hz for item in group),
        strongest=max(group, key=lambda item: item.z),
    )


# ==========================================================
# Output
# ==========================================================

def format_report(
    points: list[Point],
    reference: Point,
    reference_kind: str,
    named: list[BandResult],
    scanned: list[BandResult],
    scan_width_hz: float | None,
    band_hz: tuple[float, float],
    alpha: float,
    systematic_db: float,
    baseline_windows_hz: list[float],
) -> str:
    lines = [
        "Band power report",
        f"Created: {datetime.now().isoformat(timespec='seconds')}",
        f"alpha: {alpha:g} (Bonferroni over bands x axes of each point)   "
        f"systematic uncertainty of the reference: {systematic_db:g} dB",
        "",
        "Power in a band is compared with the same band of the reference.",
        "Excess dB = 10 log10(P / P_ref); signal = density of the excess,",
        "that is what lies under the reference in that band.",
        "",
        f"Reference ({reference_kind}): {reference.label}, "
        f"{len(reference.captures)} capture(s), {reference.packet_count} packets",
    ]
    lines += [f"  {capture.name}" for capture in reference.captures]
    for point in points:
        lines.append(
            f"Point: {point.label}, {len(point.captures)} capture(s), "
            f"{point.packet_count} packets, {point.sampling_rate_hz:.2f} Hz"
        )
        lines += [f"  {capture.name}" for capture in point.captures]

    if named:
        windows = "".join(f"  base {window:g} Hz" for window in baseline_windows_hz)
        lines += [
            "",
            "Named bands:",
            "  Spread = how much the reference captures differ from each other in",
            "  this band (std, dB): the honest size of --noise-sys-db.",
            "  Base N Hz = mean prominence over a running median of N Hz,",
            "  what a narrow-peak search with that baseline sees of the band.",
            f"{'Point':24s} {'Axis':4s} {'Band Hz':>11s} {'Excess dB':>9s} "
            f"{'z':>6s} {'thr':>5s} {'Signal ug/rtHz':>14s} {'Spread dB':>9s}{windows}  Verdict",
        ]
        for result in named:
            bases = "".join(
                f"{result.baseline_prominence_db.get(window, float('nan')):+13.2f}"
                for window in baseline_windows_hz
            )
            lines.append(
                f"{result.point[:24]:24s} {result.axis:4s} "
                f"{result.low_hz:5.2f}-{result.high_hz:<5.2f} {result.excess_db:+9.2f} "
                f"{result.z:6.1f} {result.threshold:5.2f} {result.signal_density_ug:14.1f} "
                f"{result.reference_spread_db:9.2f}{bases}  "
                f"{'above reference' if result.significant else 'not above'}"
            )

    if scan_width_hz is not None:
        structures = merge_structures(scanned)
        windows = len({(result.low_hz, result.high_hz) for result in scanned})
        spreads = np.array([
            result.reference_spread_db for result in scanned
            if np.isfinite(result.reference_spread_db)
        ])
        lines += [
            "",
            f"Scan: {scan_width_hz:g} Hz windows, half-window step, "
            f"{band_hz[0]:g}-{band_hz[1]:g} Hz, {windows} windows per axis",
        ]
        if spreads.size:
            lines.append(
                f"  reference spread between its captures: median {np.median(spreads):.2f} dB, "
                f"max {np.max(spreads):.2f} dB"
            )
        if structures:
            lines.append(
                f"{'Point':24s} {'Axis':4s} {'Range Hz':>13s}   strongest window: "
                "band, excess dB, z, signal ug/rtHz"
            )
            for structure in sorted(structures, key=lambda item: (item.point, item.axis, item.low_hz)):
                best = structure.strongest
                lines.append(
                    f"{structure.point[:24]:24s} {structure.axis:4s} "
                    f"{structure.low_hz:6.2f}-{structure.high_hz:<6.2f}   "
                    f"{best.low_hz:.2f}-{best.high_hz:.2f}, {best.excess_db:+.2f}, "
                    f"z {best.z:.1f} (thr {best.threshold:.2f}), {best.signal_density_ug:.1f}"
                )
        else:
            lines.append("  no window rises above the reference")
    return "\n".join(lines) + "\n"


def write_csv(path: Path, named: list[BandResult], scanned: list[BandResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "kind", "point", "axis", "low_hz", "high_hz", "power_g2", "power_se_g2",
            "reference_power_g2", "reference_se_g2", "excess_db", "z", "threshold",
            "signal_density_ug", "reference_spread_db", "significant",
        ])
        for kind, results in (("named", named), ("scan", scanned)):
            for result in results:
                writer.writerow([
                    kind, result.point, result.axis, f"{result.low_hz:.4f}",
                    f"{result.high_hz:.4f}", f"{result.power:.6e}", f"{result.power_se:.6e}",
                    f"{result.reference_power:.6e}", f"{result.reference_se:.6e}",
                    f"{result.excess_db:.4f}", f"{result.z:.3f}", f"{result.threshold:.3f}",
                    f"{result.signal_density_ug:.3f}", f"{result.reference_spread_db:.4f}",
                    int(result.significant),
                ])


def save_figure(
    path: Path,
    points: list[Point],
    reference: Point,
    plot_band_hz: tuple[float, float],
    named_bands: list[tuple[float, float]],
    structures: list[Structure],
    baseline_windows_hz: list[float],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    low, high = plot_band_hz
    figure, panels = plt.subplots(len(AXIS_KEYS), 1, figsize=(11, 10), sharex=True)
    styles = ("--", ":", "-.")
    point_color = {point.label: POINT_COLORS[index] for index, point in enumerate(points)}
    for panel, (axis, _) in zip(panels, AXIS_KEYS):
        for band_low, band_high in named_bands:
            panel.axvspan(band_low, band_high, color=BAND_SHADE_COLOR, alpha=0.12, lw=0)
        grid, mean = reference.mean_psd(axis)
        shown = (grid >= low) & (grid <= high)
        if reference not in points:
            panel.plot(
                grid[shown], np.sqrt(mean[shown]) * G_TO_UG,
                color=REFERENCE_COLOR, lw=1.6, label=f"опора: {reference.label}",
            )
        for point in points:
            grid, mean = point.mean_psd(axis)
            shown = (grid >= low) & (grid <= high)
            panel.plot(
                grid[shown], np.sqrt(mean[shown]) * G_TO_UG,
                color=point_color[point.label], lw=1.6,
                label=point.label + (" (опора)" if point is reference else ""),
            )
        first = points[0]
        grid, mean = first.mean_psd(axis)
        usable = grid > 0.0
        bin_width = float(grid[1] - grid[0])
        for style, window_hz in zip(styles, baseline_windows_hz):
            baseline = np.full_like(mean, np.nan)
            baseline[usable] = rolling_median(
                mean[usable], max(3, int(round(window_hz / bin_width))),
            )
            shown = (grid >= low) & (grid <= high)
            panel.plot(
                grid[shown], np.sqrt(baseline[shown]) * G_TO_UG,
                color=point_color[first.label], lw=1.2, ls=style, alpha=0.8,
                label=f"база «{first.label}»: медиана {window_hz:g} Гц",
            )
        # One row of bars per point under the curves, so points never hide each other.
        bottom, top = panel.get_ylim()
        row = 0.045 * (top - bottom)
        bottom -= row * (len(points) + 0.5)
        for index, point in enumerate(points):
            for structure in structures:
                if structure.axis == axis and structure.point == point.label:
                    panel.hlines(
                        bottom + row * (index + 0.75), structure.low_hz, structure.high_hz,
                        color=point_color[point.label], lw=4,
                    )
        panel.set_ylim(bottom, top)
        panel.set_ylabel(f"{axis}, µg/√Гц")
        panel.grid(alpha=0.25, lw=0.6)
        for side in ("top", "right"):
            panel.spines[side].set_visible(False)
    panels[0].legend(fontsize=8.5, ncol=2, frameon=False, loc="upper right")
    title = "Амплитудный спектр по точкам и база узкого поиска"
    if named_bands:
        title += "; жёлтым — заданные полосы"
    if structures:
        title += "; полосы внизу — выше опоры"
    panels[0].set_title(title, fontsize=10)
    panels[-1].set_xlabel("Частота, Гц")
    panels[-1].set_xlim(low, high)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    plt.close(figure)


# ==========================================================
# Command line
# ==========================================================

def parse_cli_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Power in frequency bands against the sensor noise floor",
    )
    parser.add_argument(
        "--point", nargs="+", action="append", required=True,
        metavar=("LABEL", "PATH"),
        help="a measuring point: its label, then capture files, glob patterns "
             "or folders; repeat for every point",
    )
    parser.add_argument(
        "--noise", nargs="+", default=None, metavar="PATH",
        help="captures holding only sensor noise: files, patterns or folders",
    )
    parser.add_argument(
        "--reference", default=None, metavar="LABEL",
        help="compare with this --point instead of --noise, e.g. the basement",
    )
    parser.add_argument(
        "--bands", nargs="+", type=parse_band, default=[], metavar="LOW-HIGH",
        help="named bands in Hz, e.g. 9.5-13 9-12",
    )
    parser.add_argument(
        "--scan", type=float, default=None, metavar="WIDTH_HZ",
        help=f"scan windows of this width across --band "
             f"(default {DEFAULT_SCAN_WIDTH_HZ:g} Hz when no --bands are given)",
    )
    parser.add_argument(
        "--band", type=float, nargs=2, default=None, metavar=("MIN_HZ", "MAX_HZ"),
        help="range of the scan (default: analysis band of config.toml)",
    )
    parser.add_argument(
        "--plot-band", type=float, nargs=2, default=None, metavar=("MIN_HZ", "MAX_HZ"),
        help=f"range of the figure (default: {DEFAULT_PLOT_BAND_HZ[0]:g}-"
             f"{DEFAULT_PLOT_BAND_HZ[1]:g} Hz, widened to cover the bands)",
    )
    parser.add_argument(
        "--nperseg", type=int, default=1024,
        help="segment length, at most one packet (default 1024)",
    )
    parser.add_argument(
        "--baseline-windows", type=float, nargs="+", default=None, metavar="HZ",
        help="running-median windows to draw and to report for named bands "
             "(default: the window of stable_spectrum.py from config.toml); "
             "pass several to compare, e.g. 5 10 15",
    )
    parser.add_argument(
        "--noise-sys-db", type=float, default=0.0,
        help="systematic uncertainty of the reference in dB (default 0)",
    )
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument(
        "--output", type=Path, default=None,
        help="folder for the report, CSV and figure "
             "(default: stable_results/band_power_<time>/)",
    )
    cli = parser.parse_args(arguments)
    for entry in cli.point:
        if len(entry) < 2:
            parser.error(f"--point {entry[0]!r} needs at least one capture after the label")
    labels = [entry[0] for entry in cli.point]
    if len(set(labels)) != len(labels):
        parser.error("point labels must be different")
    if len(labels) > len(POINT_COLORS):
        parser.error(f"at most {len(POINT_COLORS)} points; split the comparison")
    if (cli.noise is None) == (cli.reference is None):
        parser.error("give exactly one of --noise and --reference")
    if cli.reference is not None and cli.reference not in labels:
        parser.error(f"--reference {cli.reference!r} is not one of the --point labels")
    if cli.scan is None and not cli.bands:
        cli.scan = DEFAULT_SCAN_WIDTH_HZ
    if cli.baseline_windows is None:
        cli.baseline_windows = [load_settings(cli.alpha, None).baseline_window_hz]
    return cli


def main(arguments: list[str] | None = None) -> int:
    cli = parse_cli_arguments(arguments)
    try:
        points = [load_point(entry[0], entry[1:], cli.nperseg) for entry in cli.point]
        if cli.reference is not None:
            reference = next(point for point in points if point.label == cli.reference)
            reference_kind = "measuring point"
            compared = [point for point in points if point is not reference]
            if not compared:
                raise ValueError("--reference leaves no other point to compare")
        else:
            reference = load_point("шум датчика", cli.noise, cli.nperseg)
            reference_kind = "sensor noise captures"
            compared = points
        check_rates(points + ([] if cli.reference else [reference]))
        band_hz = tuple(cli.band) if cli.band else load_settings(cli.alpha, None).band_hz
        scan = scan_bands(band_hz, cli.scan) if cli.scan is not None else []
    except ValueError as error:
        print(f"error: {error}")
        return 1

    named = analyse(
        compared, reference, cli.bands, cli.alpha, cli.noise_sys_db, cli.baseline_windows,
    ) if cli.bands else []
    scanned = analyse(compared, reference, scan, cli.alpha, cli.noise_sys_db) if scan else []
    structures = merge_structures(scanned)

    output = cli.output or (
        STABLE_RESULTS_DIRECTORY / f"band_power_{datetime.now():%Y%m%d_%H%M%S}"
    )
    output.mkdir(parents=True, exist_ok=True)
    report = format_report(
        compared, reference, reference_kind, named, scanned, cli.scan, band_hz,
        cli.alpha, cli.noise_sys_db, cli.baseline_windows,
    )
    (output / "band_power_report.txt").write_text(report, encoding="utf-8")
    write_csv(output / "band_power.csv", named, scanned)
    plot_low, plot_high = cli.plot_band or DEFAULT_PLOT_BAND_HZ
    if not cli.plot_band:
        edges = [edge for band in cli.bands for edge in band] + list(band_hz if scan else [])
        if edges:
            plot_low, plot_high = min(plot_low, min(edges)), max(plot_high, max(edges))
    save_figure(
        output / "figure_band_power.png", points, reference, (plot_low, plot_high),
        cli.bands, structures, cli.baseline_windows,
    )
    print(report, end="")
    print(f"Saved band power analysis: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
