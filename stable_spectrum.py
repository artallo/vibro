"""Repeatable frequency extraction from one raw capture.

Motivation
----------
The per-run detector reports peaks that pass fixed thresholds (support,
prominence in dB). A threshold applied to a noisy spectral estimate flips
from run to run, so each measurement produces a different picture even
when the building has not changed. The instability is not in the building
and not in the clustering: it is that a peak of 1 dB means something
completely different when 16 packets were averaged than when 256 were.

This module answers a different question, which has a repeatable answer:

    which spectral peaks of this capture are larger than the uncertainty
    of the spectral estimate that produced them?

Method
------
1. One periodogram per packet (Welch, Hann, ``nperseg`` from config), so
   every packet is one independent sample of the spectrum.
2. Mean power spectral density across packets: the minimum-variance
   estimate of the true spectrum.
3. Per-bin standard error of that mean, taken from the spread across
   packets, converted to dB. No distributional assumption is made; an
   intermittent structure inflates its own error bar and is reported
   conservatively.
4. Smooth baseline: running median over frequency (reflect-padded), the
   local broadband floor.
5. Peak significance ``z = prominence_dB / standard_error_dB``. This is
   the peak height measured in units of the estimate's own noise, so it
   is comparable between captures of different length.
6. A peak is reported when ``z`` exceeds a Bonferroni-corrected Student
   quantile for the number of bins tested across all three axes, with
   ``packet_count - 1`` degrees of freedom because the error bar is
   estimated from the same packets. The threshold is derived from the
   data (bin count and record length), not tuned.
7. Split-half check: the capture is split into even and odd packets, so
   both halves span the whole recording. A real peak keeps its height in
   each half and therefore reaches ``z_threshold / sqrt(2)`` there. This
   is reported per peak as a stability flag.

What makes the output repeatable is that "nothing rises above the noise"
is itself a stable, reportable answer, and that the detection limit is
stated explicitly instead of being hidden inside a fixed dB threshold.

Usage::

    python stable_spectrum.py real_results/<capture>_raw.npz
    python stable_spectrum.py real_results/*_raw.npz --output stable_results/all
"""

from __future__ import annotations

import argparse
import csv
import sys
import tomllib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import find_peaks, welch
from scipy.stats import t

STABLE_RESULTS_DIRECTORY = Path("stable_results")
CONFIG_PATH = Path(__file__).with_name("config.toml")

DEFAULT_NPERSEG = 1024
DEFAULT_NOVERLAP = 512
DEFAULT_BAND_HZ = (0.5, 15.0)
DEFAULT_BASELINE_WINDOW_HZ = 5.0
DEFAULT_ALPHA = 0.01
DEFAULT_MIN_DISTANCE_HZ = 1.0
# Below this many packets the error bar is itself too noisy: the periodogram
# is right-skewed, the Student correction stops covering the upper tail, and
# occasional false peaks appear. Measured on the evening captures: zero false
# peaks at 64 and 256 packets, about one per ten analyses at 32.
MINIMUM_RELIABLE_PACKETS = 64

AXIS_KEYS = (("X", "x"), ("Y", "y"), ("Z", "z"))
AXIS_COLORS = {"X": "tab:blue", "Y": "tab:orange", "Z": "tab:green"}


@dataclass(frozen=True)
class SpectrumSettings:
    nperseg: int
    noverlap: int
    band_hz: tuple[float, float]
    baseline_window_hz: float
    min_distance_hz: float
    alpha: float


@dataclass(frozen=True)
class AxisSpectrum:
    axis: str
    frequencies: np.ndarray
    mean_psd: np.ndarray
    baseline_psd: np.ndarray
    prominence_db: np.ndarray
    standard_error_db: np.ndarray
    z: np.ndarray
    half_z: tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class StablePeak:
    axis: str
    frequency_hz: float
    prominence_db: float
    standard_error_db: float
    z: float
    half_z_low: float
    half_z_high: float
    persistent: bool
    band: str


@dataclass(frozen=True)
class ProbeResult:
    axis: str
    requested_hz: float
    frequency_hz: float
    prominence_db: float
    standard_error_db: float
    z: float
    upper_bound_db: float
    detection_limit_db: float
    detected: bool


@dataclass(frozen=True)
class CaptureResult:
    capture: str
    source: Path
    packet_count: int
    samples_per_packet: int
    sampling_rate_hz: float
    duration_seconds: float
    z_threshold: float
    bins_tested: int
    spectra: list[AxisSpectrum]
    peaks: list[StablePeak]
    detection_limit_db: dict[str, float]
    probes: list[ProbeResult]


# ==========================================================
# Configuration
# ==========================================================


def load_settings(alpha: float, band_hz: tuple[float, float] | None) -> SpectrumSettings:
    """Read Welch and band settings from config.toml when it is available."""
    nperseg = DEFAULT_NPERSEG
    noverlap = DEFAULT_NOVERLAP
    minimum, maximum = DEFAULT_BAND_HZ
    min_distance_hz = DEFAULT_MIN_DISTANCE_HZ
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("rb") as config_file:
            config = tomllib.load(config_file)
        welch_config = config.get("welch", {})
        nperseg = int(welch_config.get("nperseg", nperseg))
        noverlap = int(welch_config.get("noverlap", noverlap))
        bands = config.get("analysis", {}).get("bands", [])
        if bands:
            minimum = min(float(band["min_frequency"]) for band in bands)
            maximum = max(float(band["max_frequency"]) for band in bands)
            min_distance_hz = min(
                float(band.get("min_distance_hz", min_distance_hz))
                for band in bands
            )
    if band_hz is not None:
        minimum, maximum = band_hz
    return SpectrumSettings(
        nperseg=nperseg,
        noverlap=noverlap,
        band_hz=(minimum, maximum),
        baseline_window_hz=DEFAULT_BASELINE_WINDOW_HZ,
        min_distance_hz=min_distance_hz,
        alpha=alpha,
    )


def load_band_names() -> list[tuple[str, float, float]]:
    if not CONFIG_PATH.exists():
        return [("Full band", *DEFAULT_BAND_HZ)]
    with CONFIG_PATH.open("rb") as config_file:
        config = tomllib.load(config_file)
    bands = config.get("analysis", {}).get("bands", [])
    if not bands:
        return [("Full band", *DEFAULT_BAND_HZ)]
    return [
        (str(band["name"]), float(band["min_frequency"]),
         float(band["max_frequency"]))
        for band in bands
    ]


def band_name_for(frequency_hz: float, bands: list[tuple[str, float, float]]) -> str:
    for name, minimum, maximum in bands:
        if minimum <= frequency_hz <= maximum:
            return name
    return "Outside configured bands"


# ==========================================================
# Spectral estimation
# ==========================================================


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    """Running median with reflected edges.

    scipy.signal.medfilt pads with zeros, which drags the baseline down at
    the band edges and invents prominence there. Reflection keeps the
    baseline meaningful across the whole band.
    """
    if window % 2 == 0:
        window += 1
    if window <= 1 or window >= 2 * values.size:
        return np.full_like(values, float(np.median(values)))
    half = window // 2
    padded = np.pad(values, half, mode="reflect")
    strides = np.lib.stride_tricks.sliding_window_view(padded, window)
    return np.median(strides, axis=-1)


def packet_periodograms(
    signal: np.ndarray,
    sampling_rate_hz: float,
    settings: SpectrumSettings,
) -> tuple[np.ndarray, np.ndarray]:
    """One spectral estimate per packet: (frequencies, (packets, bins))."""
    nperseg = min(settings.nperseg, signal.shape[-1])
    noverlap = min(settings.noverlap, nperseg // 2)
    frequencies, psd = welch(
        signal,
        fs=sampling_rate_hz,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="density",
        axis=-1,
    )
    return frequencies, psd


def analyze_axis(
    axis: str,
    signal: np.ndarray,
    sampling_rate_hz: float,
    settings: SpectrumSettings,
) -> AxisSpectrum:
    frequencies, psd = packet_periodograms(signal, sampling_rate_hz, settings)
    selected = (
        (frequencies >= settings.band_hz[0])
        & (frequencies <= settings.band_hz[1])
    )
    frequencies = frequencies[selected]
    psd = psd[:, selected]
    packet_count = psd.shape[0]
    bin_width_hz = float(frequencies[1] - frequencies[0])
    baseline_window = max(3, int(round(settings.baseline_window_hz / bin_width_hz)))

    def profile(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean_psd = rows.mean(axis=0)
        baseline_psd = rolling_median(mean_psd, baseline_window)
        prominence_db = 10.0 * np.log10(mean_psd / baseline_psd)
        standard_error = rows.std(axis=0, ddof=1) / np.sqrt(rows.shape[0])
        standard_error_db = 10.0 * np.log10(1.0 + standard_error / mean_psd)
        return prominence_db, standard_error_db, baseline_psd

    prominence_db, standard_error_db, baseline_psd = profile(psd)
    low_half = profile(psd[0:packet_count:2])
    high_half = profile(psd[1:packet_count:2])
    return AxisSpectrum(
        axis=axis,
        frequencies=frequencies,
        mean_psd=psd.mean(axis=0),
        baseline_psd=baseline_psd,
        prominence_db=prominence_db,
        standard_error_db=standard_error_db,
        z=prominence_db / standard_error_db,
        half_z=(low_half[0] / low_half[1], high_half[0] / high_half[1]),
    )


def significance_threshold(
    bins_tested: int,
    alpha: float,
    packet_count: int,
) -> float:
    """Bonferroni-corrected Student quantile for the bins actually tested.

    ``z`` divides a mean by a standard error estimated from the same
    packets, so it follows Student's t with ``packet_count - 1`` degrees
    of freedom rather than a normal law. The difference is negligible for
    a long record and decisive for a short one: at 32 packets the normal
    quantile lets through roughly one false peak per capture.
    """
    degrees_of_freedom = max(1, packet_count - 1)
    return float(t.isf(alpha / max(1, bins_tested), degrees_of_freedom))


def find_stable_peaks(
    spectra: list[AxisSpectrum],
    z_threshold: float,
    settings: SpectrumSettings,
    bands: list[tuple[str, float, float]],
) -> list[StablePeak]:
    peaks: list[StablePeak] = []
    half_threshold = z_threshold / np.sqrt(2.0)
    for spectrum in spectra:
        bin_width_hz = float(
            spectrum.frequencies[1] - spectrum.frequencies[0]
        )
        distance = max(1, int(round(settings.min_distance_hz / bin_width_hz)))
        indices, _ = find_peaks(
            spectrum.z, height=z_threshold, distance=distance,
        )
        for index in indices:
            half_low = float(spectrum.half_z[0][index])
            half_high = float(spectrum.half_z[1][index])
            peaks.append(StablePeak(
                axis=spectrum.axis,
                frequency_hz=float(spectrum.frequencies[index]),
                prominence_db=float(spectrum.prominence_db[index]),
                standard_error_db=float(spectrum.standard_error_db[index]),
                z=float(spectrum.z[index]),
                half_z_low=half_low,
                half_z_high=half_high,
                persistent=bool(
                    half_low >= half_threshold and half_high >= half_threshold
                ),
                band=band_name_for(
                    float(spectrum.frequencies[index]), bands,
                ),
            ))
    peaks.sort(key=lambda peak: -peak.z)
    return peaks


UPPER_BOUND_SIGMA = 1.96


def probe_frequencies(
    spectra: list[AxisSpectrum],
    requested: list[float],
    z_threshold: float,
    search_radius_hz: float,
) -> list[ProbeResult]:
    """Report what sits at named frequencies, detected or not.

    A non-detection is only meaningful with a number attached, so each
    probe carries the 95% upper bound on any peak there and the prominence
    that would have been needed. A structure stronger than the upper bound
    is excluded by this record; a weaker one is not.
    """
    probes: list[ProbeResult] = []
    for spectrum in spectra:
        for requested_hz in requested:
            window = np.abs(spectrum.frequencies - requested_hz) <= search_radius_hz
            if not window.any():
                continue
            candidates = np.flatnonzero(window)
            index = int(candidates[int(np.argmax(spectrum.z[candidates]))])
            standard_error = float(spectrum.standard_error_db[index])
            probes.append(ProbeResult(
                axis=spectrum.axis,
                requested_hz=requested_hz,
                frequency_hz=float(spectrum.frequencies[index]),
                prominence_db=float(spectrum.prominence_db[index]),
                standard_error_db=standard_error,
                z=float(spectrum.z[index]),
                upper_bound_db=float(
                    spectrum.prominence_db[index]
                    + UPPER_BOUND_SIGMA * standard_error
                ),
                detection_limit_db=z_threshold * standard_error,
                detected=bool(spectrum.z[index] >= z_threshold),
            ))
    return probes


def analyze_capture(
    raw_path: Path,
    settings: SpectrumSettings,
    bands: list[tuple[str, float, float]],
    probe_hz: list[float] | None = None,
) -> CaptureResult:
    with np.load(raw_path, allow_pickle=False) as archive:
        axes = {key: np.asarray(archive[key]) for _, key in AXIS_KEYS}
        packet_fs_hz = np.asarray(archive["packet_fs_hz"])
    sampling_rate_hz = float(np.mean(packet_fs_hz))
    packet_count, samples_per_packet = axes["x"].shape
    duration_seconds = float(np.sum(samples_per_packet / packet_fs_hz))

    spectra = [
        analyze_axis(axis, axes[key], sampling_rate_hz, settings)
        for axis, key in AXIS_KEYS
    ]
    bins_tested = sum(spectrum.frequencies.size for spectrum in spectra)
    z_threshold = significance_threshold(
        bins_tested, settings.alpha, packet_count,
    )
    peaks = find_stable_peaks(spectra, z_threshold, settings, bands)
    detection_limit_db = {
        spectrum.axis: float(
            z_threshold * np.median(spectrum.standard_error_db)
        )
        for spectrum in spectra
    }
    return CaptureResult(
        capture=raw_path.stem,
        source=raw_path,
        packet_count=packet_count,
        samples_per_packet=samples_per_packet,
        sampling_rate_hz=sampling_rate_hz,
        duration_seconds=duration_seconds,
        z_threshold=z_threshold,
        bins_tested=bins_tested,
        spectra=spectra,
        peaks=peaks,
        detection_limit_db=detection_limit_db,
        probes=probe_frequencies(
            spectra,
            probe_hz or [],
            z_threshold,
            settings.min_distance_hz / 2.0,
        ),
    )


# ==========================================================
# Reporting
# ==========================================================


PEAK_FIELDS = [
    "capture", "packet_count", "duration_seconds", "sampling_rate_hz",
    "z_threshold", "axis", "band", "frequency_hz", "prominence_db",
    "standard_error_db", "z", "half_z_low", "half_z_high", "persistent",
]


def peak_rows(result: CaptureResult) -> list[dict[str, Any]]:
    return [
        {
            "capture": result.capture,
            "packet_count": result.packet_count,
            "duration_seconds": round(result.duration_seconds, 1),
            "sampling_rate_hz": round(result.sampling_rate_hz, 3),
            "z_threshold": round(result.z_threshold, 2),
            "axis": peak.axis,
            "band": peak.band,
            "frequency_hz": round(peak.frequency_hz, 3),
            "prominence_db": round(peak.prominence_db, 2),
            "standard_error_db": round(peak.standard_error_db, 3),
            "z": round(peak.z, 2),
            "half_z_low": round(peak.half_z_low, 2),
            "half_z_high": round(peak.half_z_high, 2),
            "persistent": int(peak.persistent),
        }
        for peak in result.peaks
    ]


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=PEAK_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def format_capture_report(result: CaptureResult) -> str:
    lines = [
        f"Capture: {result.capture}",
        f"Source: {result.source.resolve()}",
        f"Packets: {result.packet_count}   "
        f"Duration: {result.duration_seconds:.1f} s   "
        f"Fs: {result.sampling_rate_hz:.2f} Hz",
        f"Bins tested: {result.bins_tested} (3 axes)",
        f"Significance threshold: z >= {result.z_threshold:.2f}",
        "Detection limit (prominence needed to be significant):",
    ]
    if result.packet_count < MINIMUM_RELIABLE_PACKETS:
        lines.insert(
            3,
            f"WARNING: {result.packet_count} packets is below the "
            f"{MINIMUM_RELIABLE_PACKETS} needed for a calibrated error bar; "
            "isolated peaks here may be false.",
        )
    for axis, limit in result.detection_limit_db.items():
        lines.append(f"  {axis}: {limit:.2f} dB")
    lines.append("")
    if not result.peaks:
        strongest = max(
            (
                (spectrum.axis, float(spectrum.frequencies[int(np.argmax(spectrum.z))]),
                 float(spectrum.prominence_db[int(np.argmax(spectrum.z))]),
                 float(np.max(spectrum.z)))
                for spectrum in result.spectra
            ),
            key=lambda item: item[3],
        )
        lines.append("No frequency rises above the noise of the estimate.")
        lines.append(
            f"Strongest candidate: {strongest[0]} {strongest[1]:.2f} Hz "
            f"{strongest[2]:+.2f} dB z={strongest[3]:.1f} "
            f"(needs z >= {result.z_threshold:.2f})"
        )
        lines.append(
            "This is a stable result: a longer record is required, not a "
            "lower threshold."
        )
        return "\n".join(lines) + format_probes(result) + "\n"
    lines.append(f"Significant frequencies: {len(result.peaks)}")
    lines.append(
        "Axis  Band              Freq Hz   Prom dB   SE dB      z   "
        "half z      Persistent"
    )
    for peak in result.peaks:
        lines.append(
            f"{peak.axis:<5} {peak.band:<16} {peak.frequency_hz:8.2f}  "
            f"{peak.prominence_db:8.2f}  {peak.standard_error_db:6.3f}  "
            f"{peak.z:5.1f}   "
            f"{peak.half_z_low:4.1f}/{peak.half_z_high:<4.1f}  "
            f"{'yes' if peak.persistent else 'no':>10}"
        )
    return "\n".join(lines) + format_probes(result) + "\n"


def format_probes(result: CaptureResult) -> str:
    if not result.probes:
        return ""
    lines = [
        "",
        "Requested frequencies:",
        "Axis  Asked    Bin     Prom dB      z   Needed   95% upper   Verdict",
    ]
    for probe in result.probes:
        verdict = (
            "present" if probe.detected
            else f"absent above {probe.upper_bound_db:.2f} dB"
        )
        lines.append(
            f"{probe.axis:<5} {probe.requested_hz:6.2f} {probe.frequency_hz:6.2f}  "
            f"{probe.prominence_db:9.2f}  {probe.z:5.1f}  "
            f"{probe.detection_limit_db:7.2f}  {probe.upper_bound_db:10.2f}   "
            f"{verdict}"
        )
    return "\n" + "\n".join(lines)


def save_figure(path: Path, result: CaptureResult) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, panels = plt.subplots(
        3, 1, figsize=(13, 9), sharex=True,
    )
    for panel, spectrum in zip(panels, result.spectra):
        color = AXIS_COLORS[spectrum.axis]
        limit = result.z_threshold * spectrum.standard_error_db
        panel.fill_between(
            spectrum.frequencies, -limit, limit,
            color="0.75", alpha=0.55, lw=0,
            label=f"noise of the estimate (z < {result.z_threshold:.2f})",
        )
        panel.plot(
            spectrum.frequencies, spectrum.prominence_db,
            color=color, lw=1.4, label=f"{spectrum.axis} prominence over baseline",
        )
        panel.axhline(0.0, color="0.4", lw=0.8)
        for peak in result.peaks:
            if peak.axis != spectrum.axis:
                continue
            panel.plot(
                [peak.frequency_hz], [peak.prominence_db],
                marker="v", color="crimson", markersize=8, lw=0,
            )
            panel.annotate(
                f"{peak.frequency_hz:.2f} Hz\nz={peak.z:.1f}"
                + ("" if peak.persistent else "\n(one half only)"),
                xy=(peak.frequency_hz, peak.prominence_db),
                xytext=(0, 12), textcoords="offset points",
                ha="center", fontsize=8, color="crimson",
            )
        panel.set_ylabel(f"{spectrum.axis}\ndB over baseline")
        panel.grid(True, alpha=0.3)
        panel.legend(fontsize=8, loc="lower right", framealpha=0.9)
        extent = float(
            np.max(np.abs(np.concatenate([spectrum.prominence_db, limit])))
        )
        has_label = any(peak.axis == spectrum.axis for peak in result.peaks)
        panel.set_ylim(-1.15 * extent, extent * (1.9 if has_label else 1.15))
    panels[-1].set_xlabel("Frequency, Hz")
    verdict = (
        f"{len(result.peaks)} significant frequencies"
        if result.peaks else "no frequency above the noise of the estimate"
    )
    figure.suptitle(
        f"Stable spectrum: {result.capture}\n"
        f"{result.packet_count} packets, {result.duration_seconds:.0f} s — "
        f"{verdict}"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(path, dpi=110)
    plt.close(figure)


# ==========================================================
# Driver
# ==========================================================


def resolve_output_directory(output: Path | None, raw_paths: list[Path]) -> Path:
    if output is not None:
        return output
    if len(raw_paths) == 1:
        return STABLE_RESULTS_DIRECTORY / raw_paths[0].stem
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return STABLE_RESULTS_DIRECTORY / f"multi_{stamp}"


def run_stable_spectrum(
    raw_paths: list[Path],
    output_directory: Path,
    settings: SpectrumSettings,
    probe_hz: list[float] | None = None,
) -> Path:
    bands = load_band_names()
    output_directory.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    report_parts = [
        "Stable spectrum report",
        f"Created: {datetime.now().isoformat(timespec='seconds')}",
        f"Welch nperseg: {settings.nperseg}   "
        f"Band: {settings.band_hz[0]:g}-{settings.band_hz[1]:g} Hz   "
        f"Baseline window: {settings.baseline_window_hz:g} Hz   "
        f"alpha: {settings.alpha:g} (Bonferroni over all bins and axes)",
        "",
        "A peak is reported when its prominence over the local baseline",
        "exceeds the standard error of the spectral estimate by the",
        "corrected threshold. Half z is the same statistic on the even and",
        "odd packets of the same record.",
        "",
    ]
    for raw_path in raw_paths:
        result = analyze_capture(raw_path, settings, bands, probe_hz)
        all_rows.extend(peak_rows(result))
        report_parts.append("=" * 72)
        report_parts.append(format_capture_report(result))
        save_figure(
            output_directory / f"figure_stable_spectrum_{result.capture}.png",
            result,
        )
        verdict = (
            ", ".join(
                f"{peak.axis} {peak.frequency_hz:.2f} Hz (z={peak.z:.1f})"
                for peak in result.peaks
            )
            if result.peaks else "nothing above the noise"
        )
        print(
            f"{result.capture}: {result.packet_count} packets, "
            f"z>={result.z_threshold:.2f} -> {verdict}"
        )
    write_csv_rows(output_directory / "stable_frequencies.csv", all_rows)
    (output_directory / "stable_report.txt").write_text(
        "\n".join(report_parts), encoding="utf-8", newline="\n",
    )
    print(f"Saved stable spectrum analysis: {output_directory}")
    return output_directory


def parse_cli_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repeatable frequency extraction from raw captures",
    )
    parser.add_argument("raw_paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--alpha", type=float, default=DEFAULT_ALPHA,
        help="family-wise false-positive rate across all bins and axes",
    )
    parser.add_argument(
        "--band", type=float, nargs=2, default=None,
        metavar=("MIN_HZ", "MAX_HZ"),
        help="override the analysis band (default: from config.toml)",
    )
    parser.add_argument(
        "--probe", type=float, nargs="+", default=None, metavar="HZ",
        help="report what sits at these frequencies on every axis, "
             "including the 95%% upper bound when nothing is detected",
    )
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    cli = parse_cli_arguments(arguments)
    band = tuple(cli.band) if cli.band is not None else None
    settings = load_settings(cli.alpha, band)
    run_stable_spectrum(
        cli.raw_paths,
        resolve_output_directory(cli.output, cli.raw_paths),
        settings,
        cli.probe,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
