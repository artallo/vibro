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

Spectral resolution
-------------------
Each capture is analysed at several segment lengths (``--nperseg``,
default 1024, 2048 and 4096 samples), each written to its own
``nperseg_<n>/`` directory. A segment of one packet gives one periodogram
per packet, as above. A longer segment is cut from the packets joined into
one continuous series, with 50% overlap; the error bar then comes from the
spread across those segments, corrected for the overlap. A finer bin makes
a narrow line (a lightly damped mode) up to ``4096 / 1024`` times taller
against the noise in its bin, at the price of fewer segments and a larger
error bar. A structure wider than the bin gains nothing. Comparing the
three directories shows which kind a peak is.

Usage::

    python stable_spectrum.py real_results/<capture>_raw.npz
    python stable_spectrum.py real_results/*_raw.npz --output stable_results/all
    python stable_spectrum.py real_results/<capture>_raw.npz --nperseg 1024 4096
"""

from __future__ import annotations

import argparse
import csv
import sys
import tomllib
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import find_peaks, get_window, welch
from scipy.stats import t

STABLE_RESULTS_DIRECTORY = Path("stable_results")
CONFIG_PATH = Path(__file__).with_name("config.toml")

DEFAULT_NPERSEG = 1024
# Segment lengths analysed side by side, each into its own directory.
DEFAULT_NPERSEG_SWEEP = (1024, 2048, 4096)
# A spectral estimate needs at least this many segments to have an error
# bar at all; below it the analysis at that resolution is skipped.
MINIMUM_SEGMENTS = 4
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

# Support is counted over independent stretches of the record. The frequency
# has already been chosen by the full-record estimate, so testing it in a
# single window costs no multiple-comparison penalty and an uncorrected
# one-sided quantile is the right one. Under pure noise a frequency would
# collect support in about SUPPORT_ALPHA of the windows.
WINDOW_TARGET_COUNT = 16
MINIMUM_WINDOW_PACKETS = 16
SUPPORT_ALPHA = 0.05

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
    packet_band_power: np.ndarray
    # Number of periodograms averaged, and the number of independent ones
    # they are worth once the overlap between segments is accounted for.
    row_count: int = 0
    effective_rows: float = 0.0
    # z of a half-record relative to z of the full record, for a peak of
    # unchanged height: 1/sqrt(2) when each half holds half the rows.
    half_scale: float = float(1.0 / np.sqrt(2.0))


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
    support_windows: int = 0
    support_total: int = 0

    @property
    def support_fraction(self) -> float:
        if self.support_total == 0:
            return 0.0
        return self.support_windows / self.support_total


@dataclass(frozen=True)
class QualityReport:
    sampling_rate_spread_ppm: float
    axis_rms_g: dict[str, float]
    loud_packets: dict[str, int]
    quiet_axes: list[str]
    transients: dict[str, list[tuple[int, float]]]
    joint_steps: dict[str, list[tuple[int, float]]] = field(
        default_factory=dict,
    )

    @property
    def warnings(self) -> list[str]:
        messages = []
        for axis, items in self.joint_steps.items():
            if not items:
                continue
            shown = ", ".join(
                f"{packet}/{packet + 1} ({sigma:.0f} sigma)"
                for packet, sigma in items[:5]
            )
            more = f" and {len(items) - 5} more" if len(items) > 5 else ""
            messages.append(
                f"{axis}: step at the join of packets {shown}{more} — lost "
                "samples or a knock right at the join; segments longer than "
                "one packet straddle it"
            )
        for axis, items in self.transients.items():
            if not items:
                continue
            shown = ", ".join(
                f"{packet} ({sigma:.0f} sigma)" for packet, sigma in items[:5]
            )
            more = f" and {len(items) - 5} more" if len(items) > 5 else ""
            messages.append(
                f"{axis}: sharp transient in packet(s) {shown}{more} — "
                "a knock, or the sensor start-up when it is packet 1; it "
                "widens the error bar of that stretch but is not a frequency"
            )
        if self.sampling_rate_spread_ppm > 1000.0:
            messages.append(
                f"sampling rate varies by {self.sampling_rate_spread_ppm:.0f} "
                "ppm across packets; frequencies may be smeared"
            )
        for axis, count in self.loud_packets.items():
            if count:
                messages.append(
                    f"{axis}: {count} packet(s) far louder than the rest — "
                    "a knock or footstep, which widens the error bar"
                )
        for axis in self.quiet_axes:
            messages.append(
                f"{axis}: level far below the other axes, check the wiring"
            )
        return messages


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
    quality: QualityReport
    window_count: int
    window_packets: int
    expected_support: float
    nperseg: int = DEFAULT_NPERSEG
    bin_width_hz: float = 0.0
    row_count: int = 0
    effective_rows: float = 0.0
    window_rows: int = 0

    @property
    def segmented(self) -> bool:
        return self.nperseg > self.samples_per_packet


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


def overlap_variance_factor(nperseg: int, hop: int) -> float:
    """Variance of a mean of overlapping periodograms, relative to disjoint ones.

    Two Hann segments that overlap by half share part of their data, so
    their periodograms are correlated by ``rho``. Averaging many of them
    leaves ``1 + 2 * rho`` times the variance that the spread between
    them suggests (neighbours only; at 50% overlap nothing further apart
    overlaps). For Hann at 50% this is about 1.06.
    """
    window = get_window("hann", nperseg)
    shared = float(np.sum(window[hop:] * window[:-hop])) if hop < nperseg else 0.0
    rho = (shared / float(np.sum(window ** 2))) ** 2
    return 1.0 + 2.0 * rho


def segment_periodograms(
    signal: np.ndarray,
    sampling_rate_hz: float,
    settings: SpectrumSettings,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Periodograms at the configured resolution: (frequencies, rows, stride).

    Up to one packet long, a segment stays inside its packet and there is
    one row per packet. Longer segments are cut from the packets joined in
    order, which assumes the packets follow each other without a gap, with
    50% overlap. ``stride`` is how many rows apart two segments stop
    overlapping: 1 for packets, 2 for half-overlapping segments. The input
    must be contiguous packets of one record when it is longer than one
    packet.
    """
    samples_per_packet = signal.shape[-1]
    if settings.nperseg <= samples_per_packet:
        frequencies, psd = packet_periodograms(
            signal, sampling_rate_hz, settings,
        )
        return frequencies, psd, 1
    nperseg = settings.nperseg
    hop = nperseg // 2
    series = np.asarray(signal).reshape(-1)
    count = (series.size - nperseg) // hop + 1 if series.size >= nperseg else 0
    if count < MINIMUM_SEGMENTS:
        raise ValueError(
            f"{series.size} samples give {count} segments of {nperseg}; "
            f"at least {MINIMUM_SEGMENTS} are needed"
        )
    starts = hop * np.arange(count)
    segments = series[starts[:, None] + np.arange(nperseg)[None, :]]
    frequencies, psd = welch(
        segments,
        fs=sampling_rate_hz,
        window="hann",
        nperseg=nperseg,
        noverlap=0,
        scaling="density",
        axis=-1,
    )
    return frequencies, psd, 2


def analyze_axis(
    axis: str,
    signal: np.ndarray,
    sampling_rate_hz: float,
    settings: SpectrumSettings,
) -> AxisSpectrum:
    frequencies, psd, stride = segment_periodograms(
        signal, sampling_rate_hz, settings,
    )
    selected = (
        (frequencies >= settings.band_hz[0])
        & (frequencies <= settings.band_hz[1])
    )
    frequencies = frequencies[selected]
    psd = psd[:, selected]
    packet_count = psd.shape[0]
    bin_width_hz = float(frequencies[1] - frequencies[0])
    baseline_window = max(3, int(round(settings.baseline_window_hz / bin_width_hz)))

    variance_factor = (
        overlap_variance_factor(settings.nperseg, settings.nperseg // 2)
        if stride > 1 else 1.0
    )

    def profile(
        rows: np.ndarray,
        factor: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean_psd = rows.mean(axis=0)
        baseline_psd = rolling_median(mean_psd, baseline_window)
        prominence_db = 10.0 * np.log10(mean_psd / baseline_psd)
        standard_error = (
            rows.std(axis=0, ddof=1) * np.sqrt(factor / rows.shape[0])
        )
        standard_error_db = 10.0 * np.log10(1.0 + standard_error / mean_psd)
        return prominence_db, standard_error_db, baseline_psd

    prominence_db, standard_error_db, baseline_psd = profile(psd, variance_factor)
    # The halves take every other non-overlapping segment, so they share no
    # samples with each other and both span the whole record.
    low_rows = psd[0:packet_count:2 * stride]
    high_rows = psd[stride:packet_count:2 * stride]
    low_half = profile(low_rows, 1.0)
    high_half = profile(high_rows, 1.0)
    effective_rows = packet_count / variance_factor
    half_rows = min(low_rows.shape[0], high_rows.shape[0])
    return AxisSpectrum(
        axis=axis,
        frequencies=frequencies,
        mean_psd=psd.mean(axis=0),
        baseline_psd=baseline_psd,
        prominence_db=prominence_db,
        standard_error_db=standard_error_db,
        z=prominence_db / standard_error_db,
        half_z=(low_half[0] / low_half[1], high_half[0] / high_half[1]),
        packet_band_power=psd.sum(axis=1) * bin_width_hz,
        row_count=int(packet_count),
        effective_rows=float(effective_rows),
        half_scale=float(np.sqrt(half_rows / effective_rows)),
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
    for spectrum in spectra:
        half_threshold = z_threshold * spectrum.half_scale
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


LOUD_PACKET_RATIO = 8.0
QUIET_AXIS_RATIO = 0.05
# Peak excursion, in robust sigmas of the whole axis, that marks a packet as
# carrying a sharp transient. Gaussian noise over a quarter million samples
# tops out near 5 sigma; a knock reaches 40 and the start-up sample thousands.
TRANSIENT_SIGMA = 12.0


def transient_packets(
    signal: np.ndarray,
    sigma_threshold: float = TRANSIENT_SIGMA,
) -> list[tuple[int, float]]:
    """1-based packet numbers whose peak excursion exceeds the threshold.

    Band power misses these: a knock lasting half a second barely moves the
    in-band power of a four-second packet, yet it is exactly what a reader
    of the old time-series figure would have spotted.
    """
    median = float(np.median(signal))
    robust_std = float(np.median(np.abs(signal - median)) * 1.4826)
    if robust_std <= 0.0:
        return []
    peak_sigma = np.max(np.abs(signal - median), axis=1) / robust_std
    return [
        (int(index) + 1, float(peak_sigma[index]))
        for index in np.flatnonzero(peak_sigma > sigma_threshold)
    ]


def joint_steps(
    signal: np.ndarray,
    sigma_threshold: float = TRANSIENT_SIGMA,
) -> list[tuple[int, float]]:
    """Packet joins where the signal steps more than neighbouring samples do.

    Returns (packet, sigma) pairs, 1-based, for the join between ``packet``
    and ``packet + 1``. The step is compared with the sample-to-sample
    differences inside packets. This catches a glitch or a lost stretch
    when the signal carries slow content; in a record dominated by white
    sensor noise a lost stretch leaves no step and cannot be seen here.
    """
    if signal.ndim != 2 or signal.shape[0] < 2 or signal.shape[1] < 2:
        return []
    inner = np.diff(signal, axis=1)
    centre = float(np.median(inner))
    robust_std = float(np.median(np.abs(inner - centre)) * 1.4826)
    if robust_std <= 0.0:
        return []
    joins = signal[1:, 0] - signal[:-1, -1]
    sigma = np.abs(joins - centre) / robust_std
    return [
        (int(index) + 1, float(sigma[index]))
        for index in np.flatnonzero(sigma > sigma_threshold)
    ]


def assess_quality(
    spectra: list[AxisSpectrum],
    packet_fs_hz: np.ndarray,
    axes: dict[str, np.ndarray],
) -> QualityReport:
    """Facts about the recording itself, not about its spectrum.

    Covers what the measurement figures of main.py would otherwise be
    consulted for: did the sampling rate hold, was every axis alive, did a
    knock or a footstep dominate part of the record, did a sharp transient
    hit a packet.
    """
    mean_rate = float(np.mean(packet_fs_hz))
    spread_ppm = (
        float(np.ptp(packet_fs_hz) / mean_rate * 1.0e6) if mean_rate else 0.0
    )
    axis_rms_g = {
        spectrum.axis: float(np.sqrt(np.mean(spectrum.packet_band_power)))
        for spectrum in spectra
    }
    loud_packets = {
        spectrum.axis: int(np.sum(
            spectrum.packet_band_power
            > LOUD_PACKET_RATIO * np.median(spectrum.packet_band_power)
        ))
        for spectrum in spectra
    }
    loudest = max(axis_rms_g.values()) if axis_rms_g else 0.0
    quiet_axes = [
        axis for axis, rms in axis_rms_g.items()
        if loudest > 0.0 and rms < QUIET_AXIS_RATIO * loudest
    ]
    transients = {
        axis: transient_packets(axes[key])
        for axis, key in AXIS_KEYS
        if key in axes
    }
    steps = {
        axis: joint_steps(axes[key])
        for axis, key in AXIS_KEYS
        if key in axes
    }
    return QualityReport(
        sampling_rate_spread_ppm=spread_ppm,
        axis_rms_g=axis_rms_g,
        loud_packets=loud_packets,
        quiet_axes=quiet_axes,
        transients=transients,
        joint_steps=steps,
    )


def split_into_windows(packet_count: int) -> list[np.ndarray]:
    """Consecutive, non-overlapping stretches of the record."""
    window_packets = max(
        MINIMUM_WINDOW_PACKETS, packet_count // WINDOW_TARGET_COUNT,
    )
    window_count = packet_count // window_packets
    if window_count < 2:
        return []
    return [
        np.arange(index * window_packets, (index + 1) * window_packets)
        for index in range(window_count)
    ]


def count_support(
    window_spectra: list[AxisSpectrum],
    frequency_hz: float,
    support_threshold: float,
) -> int:
    """In how many independent windows this frequency rises above its noise.

    The frequency is fixed in advance by the full-record estimate, so each
    window is a single pre-registered test rather than a search.
    """
    support = 0
    for spectrum in window_spectra:
        index = int(np.argmin(np.abs(spectrum.frequencies - frequency_hz)))
        if spectrum.z[index] >= support_threshold:
            support += 1
    return support


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
    effective_rows = spectra[0].effective_rows
    z_threshold = significance_threshold(
        bins_tested, settings.alpha, int(round(effective_rows)),
    )
    peaks = find_stable_peaks(spectra, z_threshold, settings, bands)

    windows = split_into_windows(packet_count)
    window_packets = int(windows[0].size) if windows else 0
    window_rows = 0
    if windows and peaks:
        window_spectra = {
            axis: [
                analyze_axis(
                    axis, axes[key][indices], sampling_rate_hz, settings,
                )
                for indices in windows
            ]
            for axis, key in AXIS_KEYS
        }
        first_window = window_spectra["X"][0]
        window_rows = first_window.row_count
        support_threshold = float(t.isf(
            SUPPORT_ALPHA,
            max(1, int(round(first_window.effective_rows)) - 1),
        ))
        peaks = [
            replace(
                peak,
                support_windows=count_support(
                    window_spectra[peak.axis],
                    peak.frequency_hz,
                    support_threshold,
                ),
                support_total=len(windows),
            )
            for peak in peaks
        ]
    detection_limit_db = {
        spectrum.axis: float(
            z_threshold * np.median(spectrum.standard_error_db)
        )
        for spectrum in spectra
    }
    # The recording check looks at packets, whatever the segment length.
    if settings.nperseg == samples_per_packet:
        packet_spectra = spectra
    else:
        packet_settings = replace(
            settings,
            nperseg=samples_per_packet,
            noverlap=samples_per_packet // 2,
        )
        packet_spectra = [
            analyze_axis(axis, axes[key], sampling_rate_hz, packet_settings)
            for axis, key in AXIS_KEYS
        ]
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
        quality=assess_quality(packet_spectra, packet_fs_hz, axes),
        window_count=len(windows),
        window_packets=window_packets,
        expected_support=SUPPORT_ALPHA * len(windows),
        nperseg=settings.nperseg,
        bin_width_hz=float(
            spectra[0].frequencies[1] - spectra[0].frequencies[0]
        ),
        row_count=spectra[0].row_count,
        effective_rows=effective_rows,
        window_rows=window_rows,
    )


# ==========================================================
# Reporting
# ==========================================================


PEAK_FIELDS = [
    "capture", "nperseg", "bin_width_hz", "packet_count", "duration_seconds", "sampling_rate_hz",
    "z_threshold", "axis", "band", "frequency_hz", "prominence_db",
    "standard_error_db", "z", "half_z_low", "half_z_high", "persistent",
    "support_windows", "support_total",
]


def peak_rows(result: CaptureResult) -> list[dict[str, Any]]:
    return [
        {
            "capture": result.capture,
            "nperseg": result.nperseg,
            "bin_width_hz": round(result.bin_width_hz, 4),
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
            "support_windows": peak.support_windows,
            "support_total": peak.support_total,
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
        f"Resolution: nperseg {result.nperseg}, bin "
        f"{result.bin_width_hz:.3f} Hz, {result.row_count} "
        + (
            "segments at 50% overlap across packet joins "
            f"(worth {result.effective_rows:.0f} independent)"
            if result.segmented else "periodograms, one per packet"
        ),
        f"Bins tested: {result.bins_tested} (3 axes)",
        (
            f"Support windows: {result.window_count} x "
            f"{result.window_packets} packets"
            + (
                f" = {result.window_rows} segments each"
                if result.segmented and result.window_rows else ""
            )
            + f" (about {result.expected_support:.1f} expected by chance)"
            if result.window_count
            else "Support windows: none, record too short to split"
        ),
        f"Significance threshold: z >= {result.z_threshold:.2f}",
    ]
    lines[2] += (
        f"   Rate spread: {result.quality.sampling_rate_spread_ppm:.0f} ppm"
    )
    lines.append("")
    lines.append("Recording check:")
    lines.append(
        "  RMS in band: "
        + ", ".join(
            f"{axis} {rms:.2e} g"
            for axis, rms in result.quality.axis_rms_g.items()
        )
    )
    for message in result.quality.warnings:
        lines.append(f"  ! {message}")
    if not result.quality.warnings:
        lines.append("  no anomaly found in the recording itself")
    if result.segmented:
        lines.append(
            "  segments join consecutive packets; the file carries no packet "
            "sequence numbers, so a silently lost packet cannot be ruled out"
        )
    lines.append("")
    lines.append("Detection limit (prominence needed to be significant):")
    if result.effective_rows < MINIMUM_RELIABLE_PACKETS:
        unit = "independent segments" if result.segmented else "packets"
        lines.insert(
            3,
            f"WARNING: {result.effective_rows:.0f} {unit} is below the "
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


def noise_ceiling_psd(
    spectrum: AxisSpectrum,
    z_threshold: float,
) -> np.ndarray:
    """Power spectral density a bump has to exceed to not be sensor noise."""
    return spectrum.baseline_psd * 10.0 ** (
        z_threshold * spectrum.standard_error_db / 10.0
    )


def save_overview_figure(
    path: Path,
    result: CaptureResult,
) -> None:
    """Presentation figure: which frequencies stand out of the sensor noise.

    The shaded region is everything the record cannot distinguish from the
    noise of the instrument and of the estimate. A curve leaving that
    region is a real spectral structure; a curve inside it is not, however
    peaked it looks.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    figure, panels = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    for panel, spectrum in zip(panels, result.spectra):
        color = AXIS_COLORS[spectrum.axis]
        ceiling = noise_ceiling_psd(spectrum, result.z_threshold)
        panel.fill_between(
            spectrum.frequencies, 0.0, ceiling, color="0.86", lw=0,
        )
        panel.plot(spectrum.frequencies, ceiling, color="0.55", lw=1.0)
        panel.plot(
            spectrum.frequencies, spectrum.baseline_psd,
            color="0.45", lw=0.8, ls="--",
        )
        panel.plot(
            spectrum.frequencies, spectrum.mean_psd, color=color, lw=1.6,
        )
        axis_peaks = [
            peak for peak in result.peaks if peak.axis == spectrum.axis
        ]
        for peak in axis_peaks:
            index = int(np.argmin(
                np.abs(spectrum.frequencies - peak.frequency_hz)
            ))
            height = float(spectrum.mean_psd[index])
            panel.plot(
                [peak.frequency_hz], [height],
                marker="x", color="crimson", markersize=9,
                markeredgewidth=2, lw=0, zorder=5,
            )
            support = (
                f"{peak.support_windows}/{peak.support_total} win\n"
                if peak.support_total else ""
            )
            panel.annotate(
                f"{peak.frequency_hz:.2f} Hz\n{support}"
                f"{peak.prominence_db:.2f} dB",
                xy=(peak.frequency_hz, height),
                xytext=(0, 13), textcoords="offset points",
                ha="center", va="bottom", fontsize=8.5, color="crimson",
                zorder=6,
            )
        top = max(
            float(np.max(spectrum.mean_psd)), float(np.max(ceiling)),
        )
        panel.set_ylim(0.0, top * (1.42 if axis_peaks else 1.08))
        panel.set_ylabel(f"{spectrum.axis} axis\nPSD [g$^2$/Hz]")
        panel.grid(True, alpha=0.3)
        if not axis_peaks:
            panel.text(
                0.012, 0.94,
                "nothing rises out of the noise on this axis",
                transform=panel.transAxes, fontsize=9,
                color="0.3", va="top",
            )
    panels[-1].set_xlabel("Frequency, Hz")
    figure.legend(
        handles=[
            Patch(color="0.86"),
            Line2D([0], [0], color="0.55", lw=1.0),
            Line2D([0], [0], color="0.45", lw=0.8, ls="--"),
            Line2D([0], [0], color="0.2", lw=1.6),
            Line2D([0], [0], color="crimson", marker="x", lw=0,
                   markersize=9, markeredgewidth=2),
        ],
        labels=[
            "sensor noise: nothing here is a structure",
            "detection threshold",
            "broadband floor",
            "measured PSD (axis colour)",
            "frequency that stands out",
        ],
        loc="lower center", ncol=5, fontsize=9,
        bbox_to_anchor=(0.5, 0.0), frameon=False,
    )
    if result.peaks:
        listed = ", ".join(
            f"{peak.axis} {peak.frequency_hz:.2f} Hz"
            for peak in sorted(result.peaks, key=lambda item: item.frequency_hz)
        )
        verdict = f"Stands out of the noise: {listed}"
    else:
        verdict = (
            "Nothing stands out of the noise — a peak of "
            f"{max(result.detection_limit_db.values()):.2f} dB "
            "would have been needed"
        )
    support_note = (
        f"support counted over {result.window_count} independent windows "
        f"of {result.window_packets} packets"
        if result.window_count
        else "too short to split into independent windows, no support counted"
    )
    figure.suptitle(
        f"Dominant frequencies: {result.capture}  "
        f"[nperseg {result.nperseg}, bin {result.bin_width_hz:.3f} Hz]\n"
        f"{result.packet_count} packets, {result.duration_seconds:.0f} s; "
        f"{support_note}\n"
        f"{verdict}",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.925))
    figure.savefig(path, dpi=110)
    plt.close(figure)


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
        f"Stable spectrum: {result.capture}  "
        f"[nperseg {result.nperseg}, bin {result.bin_width_hz:.3f} Hz]\n"
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


def run_single_resolution(
    raw_paths: list[Path],
    output_directory: Path,
    settings: SpectrumSettings,
    probe_hz: list[float] | None = None,
) -> list[CaptureResult]:
    """Analyse every capture at one segment length into one directory."""
    bands = load_band_names()
    output_directory.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    results: list[CaptureResult] = []
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
        "corrected threshold. Half z is the same statistic on two",
        "interleaved halves of the same record that share no samples.",
        "",
    ]
    for raw_path in raw_paths:
        try:
            result = analyze_capture(raw_path, settings, bands, probe_hz)
        except ValueError as error:
            message = (
                f"{raw_path.stem}: skipped at nperseg {settings.nperseg}: "
                f"{error}"
            )
            print(message)
            report_parts.append("=" * 72)
            report_parts.append(message + "\n")
            continue
        results.append(result)
        all_rows.extend(peak_rows(result))
        report_parts.append("=" * 72)
        report_parts.append(format_capture_report(result))
        save_figure(
            output_directory / f"figure_stable_spectrum_{result.capture}.png",
            result,
        )
        save_overview_figure(
            output_directory / f"figure_dominant_{result.capture}.png",
            result,
        )
        print(
            f"{result.capture} [nperseg {result.nperseg}]: "
            f"{result.row_count} rows, z>={result.z_threshold:.2f} -> "
            f"{peak_summary(result)}"
        )
    write_csv_rows(output_directory / "stable_frequencies.csv", all_rows)
    (output_directory / "stable_report.txt").write_text(
        "\n".join(report_parts), encoding="utf-8", newline="\n",
    )
    return results


def peak_summary(result: CaptureResult) -> str:
    if not result.peaks:
        return "nothing above the noise"
    return ", ".join(
        f"{peak.axis} {peak.frequency_hz:.2f} Hz (z={peak.z:.1f})"
        for peak in result.peaks
    )


def resolution_directory_name(nperseg: int) -> str:
    return f"nperseg_{nperseg}"


def format_comparison(
    results_by_nperseg: dict[int, list[CaptureResult]],
) -> str:
    """One table per capture: what each resolution found."""
    captures: dict[str, dict[int, CaptureResult]] = {}
    for nperseg, results in results_by_nperseg.items():
        for result in results:
            captures.setdefault(result.capture, {})[nperseg] = result
    lines = [
        "Resolution comparison",
        f"Created: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "A narrow line (a lightly damped mode) grows taller as the bin",
        "narrows, until the bin is narrower than the line. A structure wider",
        "than the bin keeps its height, while the error bar grows because",
        "fewer segments are averaged. Detection limits are in dB of",
        "prominence at that resolution.",
        "",
    ]
    for capture, by_nperseg in captures.items():
        lines.append("=" * 72)
        lines.append(f"Capture: {capture}")
        lines.append(
            "nperseg   bin Hz  segments  z thr   limit X/Y/Z dB     "
            "significant frequencies"
        )
        for nperseg in sorted(by_nperseg):
            result = by_nperseg[nperseg]
            limits = "/".join(
                f"{result.detection_limit_db[axis]:.2f}"
                for axis, _ in AXIS_KEYS
            )
            lines.append(
                f"{nperseg:7d}  {result.bin_width_hz:7.3f}  "
                f"{result.row_count:8d}  {result.z_threshold:5.2f}   "
                f"{limits:<17}  {peak_summary(result)}"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def save_comparison_figure(
    path: Path,
    by_nperseg: dict[int, CaptureResult],
) -> None:
    """Measured PSD of one capture at every resolution, overlaid per axis.

    The density is in g/sqrt(Hz), which does not depend on the bin width,
    so a broad structure and the sensor floor lie on top of each other at
    every resolution; only narrow lines rise with finer bins.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(by_nperseg)
    shades = plt.cm.viridis(np.linspace(0.0, 0.8, len(ordered)))
    figure, panels = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    for axis_index, (panel, (axis, _)) in enumerate(zip(panels, AXIS_KEYS)):
        for shade, nperseg in zip(shades, ordered):
            result = by_nperseg[nperseg]
            spectrum = result.spectra[axis_index]
            panel.plot(
                spectrum.frequencies,
                np.sqrt(spectrum.mean_psd) * 1.0e6,
                color=shade, lw=1.1 if nperseg == ordered[0] else 0.9,
                label=(
                    f"nperseg {nperseg}, bin {result.bin_width_hz:.3f} Hz, "
                    f"limit {result.detection_limit_db[axis]:.2f} dB"
                ),
            )
            for peak in result.peaks:
                if peak.axis != axis:
                    continue
                index = int(np.argmin(
                    np.abs(spectrum.frequencies - peak.frequency_hz)
                ))
                panel.plot(
                    [peak.frequency_hz],
                    [np.sqrt(spectrum.mean_psd[index]) * 1.0e6],
                    marker="x", color=shade, markersize=9,
                    markeredgewidth=2, lw=0,
                )
        panel.set_ylabel(f"{axis} axis\nASD [µg/√Hz]")
        panel.grid(True, alpha=0.3)
        panel.legend(fontsize=8, loc="upper right", framealpha=0.9)
    panels[-1].set_xlabel("Frequency, Hz")
    capture = by_nperseg[ordered[0]].capture
    figure.suptitle(
        f"Resolution comparison: {capture}\n"
        "crosses mark frequencies significant at that resolution; "
        "only narrow lines rise with finer bins",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(path, dpi=110)
    plt.close(figure)


def run_stable_spectrum(
    raw_paths: list[Path],
    output_directory: Path,
    settings: SpectrumSettings,
    probe_hz: list[float] | None = None,
    nperseg_values: list[int] | None = None,
) -> Path:
    """Analyse captures, once per segment length.

    Without ``nperseg_values`` the configured segment length is written
    straight into ``output_directory``. With them, each length gets its own
    ``nperseg_<n>/`` directory, and ``comparison.txt`` plus one overlay
    figure per capture are written next to them.
    """
    if not nperseg_values:
        run_single_resolution(raw_paths, output_directory, settings, probe_hz)
        print(f"Saved stable spectrum analysis: {output_directory}")
        return output_directory
    output_directory.mkdir(parents=True, exist_ok=True)
    results_by_nperseg: dict[int, list[CaptureResult]] = {}
    for nperseg in dict.fromkeys(nperseg_values):
        resolution_settings = replace(
            settings, nperseg=int(nperseg), noverlap=int(nperseg) // 2,
        )
        results_by_nperseg[int(nperseg)] = run_single_resolution(
            raw_paths,
            output_directory / resolution_directory_name(int(nperseg)),
            resolution_settings,
            probe_hz,
        )
    (output_directory / "comparison.txt").write_text(
        format_comparison(results_by_nperseg), encoding="utf-8", newline="\n",
    )
    per_capture: dict[str, dict[int, CaptureResult]] = {}
    for nperseg, results in results_by_nperseg.items():
        for result in results:
            per_capture.setdefault(result.capture, {})[nperseg] = result
    for capture, by_nperseg in per_capture.items():
        save_comparison_figure(
            output_directory / f"figure_compare_{capture}.png", by_nperseg,
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
        "--nperseg", type=int, nargs="+",
        default=list(DEFAULT_NPERSEG_SWEEP), metavar="SAMPLES",
        help="segment lengths to analyse, each into its own nperseg_<n> "
             "directory (default: 1024 2048 4096)",
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
        cli.nperseg,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
