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
   local broadband floor. Its width (10 Hz, ``[stable_spectrum]`` in
   config.toml) is the widest structure whose top still stands above it.
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

A folder on the command line stands for one measuring point: every
``*_raw.npz`` in it (not in its subfolders) is a run of that point.

Usage::

    python stable_spectrum.py real_results/<capture>_raw.npz
    python stable_spectrum.py "tumen_results/260913 Тюмень, 10k"
    python stable_spectrum.py "tumen_results/260913 Тюмень, 10k" --pool
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

# Segment lengths analysed side by side, each into its own directory: what
# overview_figure.py and false_alarm_check.py compare, and what
# stable_spectrum.py runs with --nperseg 1024 2048 4096.
DEFAULT_NPERSEG_SWEEP = (1024, 2048, 4096)
# What stable_spectrum.py analyses when --nperseg is not given. Building
# modes found so far are about as wide as a 2048 bin (0.12 Hz): z is highest
# there, 1024 is too coarse and 4096 only adds error. Overridden by
# [stable_spectrum] nperseg in config.toml.
DEFAULT_NPERSEG_RUN = (2048,)
# A spectral estimate needs at least this many segments to have an error
# bar at all; below it the analysis at that resolution is skipped.
MINIMUM_SEGMENTS = 4
DEFAULT_BAND_HZ = (0.2, 15.0)
# Width of the running-median baseline. A structure wider than about half
# of it lifts its own baseline. At 5 Hz the top of a 3-4 Hz wide hump lost
# about a dB and flickered across runs and resolutions; 10 Hz keeps it and
# still follows the broadband floor. Overridden by [stable_spectrum] in
# config.toml or --baseline-window.
DEFAULT_BASELINE_WINDOW_HZ = 10.0
DEFAULT_ALPHA = 0.01
# How far from a requested frequency the probe looks for its peak. Wide
# enough for a mode that moves a little between runs, floors or
# instruments (a few tenths of a hertz at most so far), narrow enough that a
# stronger neighbour 0.5 Hz away does not answer for it.
DEFAULT_PROBE_TOLERANCE_HZ = 0.2
DEFAULT_MIN_DISTANCE_HZ = 1.0
# With a separation test, peaks are first looked for this many bins apart.
SEPARATION_MIN_BINS = 2
# A weaker peak within min_distance_hz of a stronger one stays when the dip
# between them is deeper than this many standard errors. Overridden by
# [stable_spectrum] separation_sigma in config.toml or --separation-sigma;
# 0 merges every pair closer than min_distance_hz, as before 2026-10-02.
DEFAULT_SEPARATION_SIGMA = 2.0
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
    # 0: peaks closer than min_distance_hz merge, the stronger z stays.
    # Above 0: a weaker peak inside min_distance_hz of a stronger one stays
    # too when the dip between them is deeper than this many standard
    # errors.
    separation_sigma: float = 0.0


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
    # The same z profile extended past both band edges where the spectrum
    # has bins, and the index of the first in-band bin inside it. Peaks are
    # searched on this wider profile so a line in the edge bin of the band
    # is judged against its real neighbour instead of a wall.
    context_frequencies: np.ndarray | None = None
    context_z: np.ndarray | None = None
    context_mean_psd: np.ndarray | None = None
    band_offset: int = 0


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
    # frequency_hz is the top of the peak found between bins; this keeps the
    # centre of the bin that held the maximum.
    bin_frequency_hz: float = 0.0
    # Pooled records only: in how many of the pooled runs the blind search
    # of that run alone finds this frequency, on the same axis.
    runs_found: int = 0
    runs_total: int = 0

    @property
    def runs_label(self) -> str:
        return f"{self.runs_found}/{self.runs_total}" if self.runs_total > 1 else ""

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
    source: str = ""


@dataclass(frozen=True)
class ProbeRequest:
    """A frequency chosen before this record: from another run, floor or instrument."""

    frequency_hz: float
    axis: str | None = None  # None: look on every axis
    source: str = ""

    def label(self) -> str:
        if self.axis:
            return f"{self.axis}:{self.frequency_hz:g}"
        return f"{self.frequency_hz:g}"


@dataclass(frozen=True)
class ProbePlan:
    """What to look at besides the blind search, and how strictly.

    ``requests`` come from the command line; ``candidates`` are the peaks a
    search on another run found, to be confirmed here. Both are judged with
    a threshold that counts only the bins inside their windows, which is
    honest only because the frequencies were not taken from this record.
    ``candidate_captures`` names the records the candidates came from, so a
    record is never asked to confirm its own findings.
    """

    requests: tuple[ProbeRequest, ...] = ()
    candidates: tuple[ProbeRequest, ...] = ()
    candidate_captures: frozenset[str] = frozenset()
    candidate_sources: tuple[str, ...] = ()
    tolerance_hz: float = DEFAULT_PROBE_TOLERANCE_HZ
    alpha: float = DEFAULT_ALPHA


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
    nperseg: int
    bin_width_hz: float = 0.0
    row_count: int = 0
    effective_rows: float = 0.0
    window_rows: int = 0
    edge_warnings: list[str] = field(default_factory=list)
    startup_dropped: bool = False
    probe_plan: ProbePlan = field(default_factory=ProbePlan)
    probe_threshold: float = 0.0
    probe_bins: int = 0
    confirmations: list[ProbeResult] = field(default_factory=list)
    confirm_threshold: float = 0.0
    confirm_bins: int = 0
    confirm_skipped: str = ""
    # Pooled records: the runs, their packet counts and their own checks.
    members: tuple[str, ...] = ()
    member_packets: tuple[int, ...] = ()
    member_quality: tuple[tuple[str, "QualityReport"], ...] = ()
    pool_warnings: tuple[str, ...] = ()

    @property
    def pooled(self) -> bool:
        return len(self.members) > 1

    @property
    def segmented(self) -> bool:
        return self.nperseg > self.samples_per_packet


# ==========================================================
# Configuration
# ==========================================================


def load_settings(
    alpha: float,
    band_hz: tuple[float, float] | None,
    baseline_window_hz: float | None = None,
    min_distance_hz: float | None = None,
    separation_sigma: float | None = None,
) -> SpectrumSettings:
    """Read band and baseline settings from config.toml when it is available.

    The segment length is the first of [stable_spectrum] nperseg, with half
    of it as overlap. stable_spectrum.py replaces it with each length it
    analyses, and the other scripts set their own; it only stays for a
    caller that analyses these settings as they are. [welch] of config.toml
    is the mode of the old detector in main.py and is not read here.
    """
    nperseg = load_default_nperseg()[0]
    noverlap = nperseg // 2
    minimum, maximum = DEFAULT_BAND_HZ
    min_distance_hz_config = DEFAULT_MIN_DISTANCE_HZ
    configured_window_hz = DEFAULT_BASELINE_WINDOW_HZ
    configured_sigma = DEFAULT_SEPARATION_SIGMA
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("rb") as config_file:
            config = tomllib.load(config_file)
        bands = config.get("analysis", {}).get("bands", [])
        if bands:
            minimum = min(float(band["min_frequency"]) for band in bands)
            maximum = max(float(band["max_frequency"]) for band in bands)
            min_distance_hz_config = min(
                float(band.get("min_distance_hz", min_distance_hz_config))
                for band in bands
            )
        own = config.get("stable_spectrum", {})
        configured_window_hz = float(own.get("baseline_window_hz", configured_window_hz))
        # Its own key, so that the old detector keeps the min_distance_hz
        # of [[analysis.bands]].
        configured_distance_hz = own.get("min_distance_hz")
        if configured_distance_hz is not None:
            min_distance_hz_config = float(configured_distance_hz)
        configured_sigma = float(own.get("separation_sigma", configured_sigma))
    if band_hz is not None:
        minimum, maximum = band_hz
    if baseline_window_hz is not None:
        configured_window_hz = baseline_window_hz
    if configured_window_hz <= 0.0:
        raise ValueError("the baseline window must be positive")
    if min_distance_hz is not None:
        min_distance_hz_config = min_distance_hz
    if separation_sigma is not None:
        configured_sigma = separation_sigma
    if min_distance_hz_config < 0.0 or configured_sigma < 0.0:
        raise ValueError("min distance and separation sigma must not be negative")
    return SpectrumSettings(
        nperseg=nperseg,
        noverlap=noverlap,
        band_hz=(minimum, maximum),
        baseline_window_hz=configured_window_hz,
        min_distance_hz=min_distance_hz_config,
        alpha=alpha,
        separation_sigma=configured_sigma,
    )


def load_default_nperseg() -> list[int]:
    """Segment lengths stable_spectrum.py analyses when none are given."""
    values = DEFAULT_NPERSEG_RUN
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("rb") as config_file:
            config = tomllib.load(config_file)
        values = config.get("stable_spectrum", {}).get("nperseg", values)
    if isinstance(values, int):
        values = [values]
    values = [int(value) for value in values]
    if not values or any(value < 16 for value in values):
        raise ValueError(f"[stable_spectrum] nperseg must list segment lengths, got {values}")
    return values


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
    groups: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Periodograms at the configured resolution: (frequencies, rows, stride).

    Up to one packet long, a segment stays inside its packet and there is
    one row per packet. Longer segments are cut from the packets joined in
    order, which assumes the packets follow each other without a gap, with
    50% overlap. ``stride`` is how many rows apart two segments stop
    overlapping: 1 for packets, 2 for half-overlapping segments.

    ``groups`` gives the packet count of each record when several records
    are stacked: each record is cut on its own and the segments of all of
    them are pooled, so no segment straddles the pause between two records.
    """
    samples_per_packet = signal.shape[-1]
    if settings.nperseg <= samples_per_packet:
        frequencies, psd = packet_periodograms(
            signal, sampling_rate_hz, settings,
        )
        return frequencies, psd, 1
    nperseg = settings.nperseg
    hop = nperseg // 2
    signal = np.asarray(signal)
    sizes = groups if groups else [signal.shape[0]]
    if sum(sizes) != signal.shape[0]:
        raise ValueError(
            f"groups add up to {sum(sizes)} packets, the signal has {signal.shape[0]}"
        )
    all_segments = []
    start_packet = 0
    for size in sizes:
        series = signal[start_packet:start_packet + size].reshape(-1)
        start_packet += size
        count = (series.size - nperseg) // hop + 1 if series.size >= nperseg else 0
        if count > 0:
            starts = hop * np.arange(count)
            all_segments.append(series[starts[:, None] + np.arange(nperseg)[None, :]])
    total = sum(block.shape[0] for block in all_segments)
    if total < MINIMUM_SEGMENTS:
        raise ValueError(
            f"{signal.size} samples give {total} segments of {nperseg}; "
            f"at least {MINIMUM_SEGMENTS} are needed"
        )
    frequencies, psd = welch(
        np.vstack(all_segments),
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
    groups: list[int] | None = None,
) -> AxisSpectrum:
    all_frequencies, all_psd, stride = segment_periodograms(
        signal, sampling_rate_hz, settings, groups,
    )
    bin_width_hz = float(all_frequencies[1] - all_frequencies[0])
    baseline_window = max(3, int(round(settings.baseline_window_hz / bin_width_hz)))
    band_indices = np.flatnonzero(
        (all_frequencies >= settings.band_hz[0])
        & (all_frequencies <= settings.band_hz[1])
    )
    # Context: half a baseline window of real bins past each band edge, so
    # that the baseline and the peak search near the edges rest on data
    # rather than on a mirror image. The DC bin carries only what the
    # detrending left and is never used as a neighbour.
    margin = baseline_window // 2
    context_start = min(
        int(band_indices[0]),
        max(1, int(band_indices[0]) - margin),
    )
    context_stop = min(all_frequencies.size, int(band_indices[-1]) + 1 + margin)
    band_offset = int(band_indices[0]) - context_start
    band_slice = slice(band_offset, band_offset + band_indices.size)
    context_frequencies = all_frequencies[context_start:context_stop]
    psd = all_psd[:, context_start:context_stop]
    packet_count = psd.shape[0]

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
    context_z = prominence_db / standard_error_db
    return AxisSpectrum(
        axis=axis,
        frequencies=context_frequencies[band_slice],
        mean_psd=psd.mean(axis=0)[band_slice],
        baseline_psd=baseline_psd[band_slice],
        prominence_db=prominence_db[band_slice],
        standard_error_db=standard_error_db[band_slice],
        z=context_z[band_slice],
        half_z=(
            (low_half[0] / low_half[1])[band_slice],
            (high_half[0] / high_half[1])[band_slice],
        ),
        packet_band_power=psd[:, band_slice].sum(axis=1) * bin_width_hz,
        row_count=int(packet_count),
        effective_rows=float(effective_rows),
        half_scale=float(np.sqrt(half_rows / effective_rows)),
        context_frequencies=context_frequencies,
        context_z=context_z,
        context_mean_psd=psd.mean(axis=0),
        band_offset=band_offset,
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


def refine_peak(
    frequencies: np.ndarray,
    mean_psd: np.ndarray,
    index: int,
) -> float:
    """Frequency of a peak between bins.

    A parabola through the log PSD of the peak bin and its two neighbours
    puts the top of a Hann-windowed peak to a small fraction of a bin.
    The reported bin centre can be off by up to half a bin, 0.12 Hz at
    nperseg 1024, and a line lying between two bins flips from one to the
    other between runs; the parabola top does neither. The shift is kept
    within half a bin, and a peak without two neighbours is left as it is.
    """
    frequency = float(frequencies[index])
    if index <= 0 or index >= mean_psd.size - 1:
        return frequency
    tiny = np.finfo(float).tiny
    left, centre, right = 10.0 * np.log10(
        np.maximum(mean_psd[index - 1:index + 2], tiny)
    )
    curvature = left - 2.0 * centre + right
    if not curvature < 0.0:
        return frequency
    offset = float(np.clip(0.5 * (left - right) / curvature, -0.5, 0.5))
    step = float(frequencies[1] - frequencies[0])
    return frequency + offset * step


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
        window = max(1, int(round(settings.min_distance_hz / bin_width_hz)))
        separating = settings.separation_sigma > 0.0
        distance = min(window, SEPARATION_MIN_BINS) if separating else window
        # Search the profile that reaches past the band edges, then keep the
        # peaks whose top lies inside the band. A peak in the edge bin is
        # then compared with the real bin beyond the edge.
        if spectrum.context_z is not None:
            search_z, offset = spectrum.context_z, spectrum.band_offset
        else:
            search_z, offset = spectrum.z, 0
        found, _ = find_peaks(search_z, height=z_threshold, distance=distance)
        indices = found - offset
        indices = indices[(indices >= 0) & (indices < spectrum.z.size)]
        if separating:
            indices = separate_close_peaks(
                spectrum, indices, window, settings.separation_sigma,
            )
        for index in indices:
            half_low = float(spectrum.half_z[0][index])
            half_high = float(spectrum.half_z[1][index])
            if spectrum.context_mean_psd is not None:
                refined_hz = refine_peak(
                    spectrum.context_frequencies,
                    spectrum.context_mean_psd,
                    int(index) + offset,
                )
            else:
                refined_hz = refine_peak(
                    spectrum.frequencies, spectrum.mean_psd, int(index),
                )
            peaks.append(StablePeak(
                axis=spectrum.axis,
                frequency_hz=refined_hz,
                bin_frequency_hz=float(spectrum.frequencies[index]),
                prominence_db=float(spectrum.prominence_db[index]),
                standard_error_db=float(spectrum.standard_error_db[index]),
                z=float(spectrum.z[index]),
                half_z_low=half_low,
                half_z_high=half_high,
                persistent=bool(
                    half_low >= half_threshold and half_high >= half_threshold
                ),
                band=band_name_for(refined_hz, bands),
            ))
    peaks.sort(key=lambda peak: -peak.z)
    return peaks


def separate_close_peaks(
    spectrum: AxisSpectrum,
    indices: np.ndarray,
    window_bins: int,
    sigma: float,
) -> np.ndarray:
    """Keep a weaker peak near a stronger one only behind a real dip.

    Peaks are taken by z, strongest first. A peak within ``window_bins`` of
    one already kept stays only when the lowest prominence between the two
    lies below its own prominence by more than ``sigma`` standard errors of
    that difference. A wiggle on the flank or on a ragged top has no such
    dip and goes; a second mode beside a first one keeps it.
    """
    kept: list[int] = []
    for index in sorted((int(item) for item in indices), key=lambda item: -spectrum.z[item]):
        separate = True
        for other in kept:
            # As for find_peaks, peaks exactly window_bins apart are already apart.
            if abs(index - other) >= window_bins:
                continue
            low, high = sorted((index, other))
            if high - low < 2:
                separate = False
                break
            saddle = low + 1 + int(np.argmin(spectrum.prominence_db[low + 1:high]))
            dip = float(spectrum.prominence_db[index] - spectrum.prominence_db[saddle])
            error = float(np.hypot(
                spectrum.standard_error_db[index], spectrum.standard_error_db[saddle],
            ))
            if dip < sigma * error:
                separate = False
                break
        if separate:
            kept.append(index)
    return np.array(sorted(kept), dtype=int)


def band_edge_warnings(
    spectra: list[AxisSpectrum],
    z_threshold: float,
) -> list[str]:
    """Significant structure the band cannot report as a peak.

    Two cases. The edge bin of the band may be the last bin the spectrum
    has on that side, so there is no neighbour to show whether the top lies
    at the edge or beyond it. Or a peak may stand just outside the band, in
    the context bins. Either way the reader is told to move the band edge
    instead of being left with silence.
    """
    messages = []
    for spectrum in spectra:
        if spectrum.context_z is None or spectrum.context_frequencies is None:
            continue
        context_z = spectrum.context_z
        context_frequencies = spectrum.context_frequencies
        first = spectrum.band_offset
        last = first + spectrum.z.size - 1
        if (
            first == 0
            and context_z[0] >= z_threshold
            and (context_z.size < 2 or context_z[0] > context_z[1])
        ):
            messages.append(
                f"{spectrum.axis}: {context_frequencies[0]:.2f} Hz, the lowest "
                f"bin available, stands at z={context_z[0]:.1f}; nothing below "
                "it shows whether the top lies there or lower down"
            )
        if (
            last == context_z.size - 1
            and context_z[-1] >= z_threshold
            and (context_z.size < 2 or context_z[-1] > context_z[-2])
        ):
            messages.append(
                f"{spectrum.axis}: {context_frequencies[-1]:.2f} Hz, the highest "
                f"bin available, stands at z={context_z[-1]:.1f}; nothing above "
                "it shows whether the top lies there or higher up"
            )
        outside, _ = find_peaks(context_z, height=z_threshold)
        for index in outside:
            if first <= index <= last:
                continue
            side = "below" if index < first else "above"
            messages.append(
                f"{spectrum.axis}: a peak at {context_frequencies[index]:.2f} Hz "
                f"(z={context_z[index]:.1f}) lies just {side} the band; widen "
                "--band to include it"
            )
    return messages


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


def drop_startup_packet(
    axes: dict[str, np.ndarray],
    packet_fs_hz: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray, bool]:
    """Leave out packet 1 when it carries a sharp transient.

    The sensor starts every recording from zero, so packet 1 holds a step
    of a whole g on Z. Averaged into the spectrum it lifts the lowest bin
    by 12-21 dB and fakes a structure there. The recording check still
    reports the transient; only the spectra are computed without it. A
    knock that happens to fall into packet 1 is left out the same way,
    which costs one packet out of hundreds.
    """
    if axes["x"].shape[0] < 2:
        return axes, packet_fs_hz, False
    started = any(
        any(packet == 1 for packet, _ in transient_packets(axes[key]))
        for _, key in AXIS_KEYS
        if key in axes
    )
    if not started:
        return axes, packet_fs_hz, False
    return (
        {key: signal[1:] for key, signal in axes.items()},
        packet_fs_hz[1:],
        True,
    )


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


def as_request(item: ProbeRequest | float) -> ProbeRequest:
    return item if isinstance(item, ProbeRequest) else ProbeRequest(float(item))


def spectrum_bin_width(spectrum: AxisSpectrum) -> float:
    if spectrum.frequencies.size < 2:
        return 0.0
    return float(spectrum.frequencies[1] - spectrum.frequencies[0])


def probe_window(
    spectrum: AxisSpectrum,
    request: ProbeRequest,
    search_radius_hz: float,
) -> np.ndarray:
    """Bins a probe may answer from: within the radius, at least the nearest bin.

    A request on another axis, or outside the analysed band, gets no bins.
    """
    if request.axis is not None and request.axis != spectrum.axis:
        return np.array([], dtype=int)
    if spectrum.frequencies.size == 0:
        return np.array([], dtype=int)
    distance = np.abs(spectrum.frequencies - request.frequency_hz)
    nearest = int(np.argmin(distance))
    if distance[nearest] > search_radius_hz + spectrum_bin_width(spectrum):
        return np.array([], dtype=int)
    window = np.flatnonzero(distance <= search_radius_hz)
    return window if window.size else np.array([nearest])


def probe_threshold(
    spectra: list[AxisSpectrum],
    requests: list[ProbeRequest | float],
    search_radius_hz: float,
    alpha: float,
    rows: int,
) -> tuple[float, int]:
    """Threshold for frequencies chosen before the record, and the bins it covers.

    Only the bins inside the probe windows can produce a false answer, so
    the Bonferroni correction counts those and not the whole band. This is
    honest only when the frequencies were not picked from this record.
    """
    bins = sum(
        probe_window(spectrum, as_request(item), search_radius_hz).size
        for spectrum in spectra
        for item in requests
    )
    if bins == 0:
        return 0.0, 0
    return significance_threshold(bins, alpha, rows), bins


def probe_frequencies(
    spectra: list[AxisSpectrum],
    requested: list[ProbeRequest | float],
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
        for item in requested:
            request = as_request(item)
            requested_hz = request.frequency_hz
            candidates = probe_window(spectrum, request, search_radius_hz)
            if candidates.size == 0:
                continue
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
                source=request.source,
            ))
    return probes


def parse_probe(text: str) -> ProbeRequest:
    """``2.88`` looks on every axis, ``Y:2.88`` only on Y."""
    axis = None
    value = text.strip()
    if ":" in value:
        axis, value = (part.strip() for part in value.split(":", 1))
        axis = axis.upper()
        if axis not in {name for name, _ in AXIS_KEYS}:
            raise argparse.ArgumentTypeError(
                f"probe '{text}': the axis must be X, Y or Z"
            )
    try:
        frequency_hz = float(value.replace(",", "."))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"probe '{text}' is not a frequency like 2.88 or Y:2.88"
        ) from error
    if frequency_hz <= 0.0:
        raise argparse.ArgumentTypeError(f"probe '{text}' must be positive")
    return ProbeRequest(frequency_hz, axis)


def load_candidates(
    paths: list[Path],
    tolerance_hz: float = DEFAULT_PROBE_TOLERANCE_HZ,
) -> tuple[tuple[ProbeRequest, ...], frozenset[str], tuple[str, ...]]:
    """Peaks found by a search on other runs, to be confirmed on this one.

    Reads ``stable_frequencies.csv`` files, or every such file under a
    folder. Peaks of one axis closer than the tolerance count as one
    candidate, at the frequency of the strongest. Returns the candidates,
    the names of the records they came from and the files read.
    """
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("stable_frequencies.csv")))
        elif path.exists():
            files.append(path)
    if not files:
        listed = " ".join(str(path) for path in paths)
        raise ValueError(f"no stable_frequencies.csv found in {listed}")
    found: list[tuple[str, float, float, str]] = []
    captures: set[str] = set()
    for file in files:
        with file.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                captures.add(row["capture"])
                nperseg = row.get("nperseg") or "?"
                found.append((
                    row["axis"], float(row["frequency_hz"]), float(row["z"]),
                    f"{row['capture']} n{nperseg}",
                ))
    kept: list[tuple[str, float, float, str]] = []
    for axis, frequency, z, source in sorted(found, key=lambda item: -item[2]):
        if any(
            other[0] == axis and abs(other[1] - frequency) <= tolerance_hz
            for other in kept
        ):
            continue
        kept.append((axis, frequency, z, source))
    requests = tuple(
        ProbeRequest(frequency, axis, f"{source} z={z:.1f}")
        for axis, frequency, z, source in sorted(kept, key=lambda item: (item[0], item[1]))
    )
    return requests, frozenset(captures), tuple(str(file) for file in files)


@dataclass(frozen=True)
class LoadedCapture:
    name: str
    source: Path
    axes: dict[str, np.ndarray]  # start-up packet left out when it had a transient
    packet_fs_hz: np.ndarray
    recorded_axes: dict[str, np.ndarray]
    recorded_fs_hz: np.ndarray
    startup_dropped: bool


def load_capture(raw_path: Path) -> LoadedCapture:
    with np.load(raw_path, allow_pickle=False) as archive:
        recorded_axes = {key: np.asarray(archive[key]) for _, key in AXIS_KEYS}
        recorded_fs_hz = np.asarray(archive["packet_fs_hz"])
    axes, packet_fs_hz, startup_dropped = drop_startup_packet(
        recorded_axes, recorded_fs_hz,
    )
    return LoadedCapture(
        name=raw_path.stem,
        source=raw_path,
        axes=axes,
        packet_fs_hz=packet_fs_hz,
        recorded_axes=recorded_axes,
        recorded_fs_hz=recorded_fs_hz,
        startup_dropped=startup_dropped,
    )


# Records pooled together must share one frequency grid.
MAX_POOL_RATE_MISMATCH = 0.005
# Runs of one sensor placement differ in mean level by at most a few times
# the noise of one sample (up to 7x on all records so far); moving the
# sensor tilts it and shifts X and Y by 70-170x. A spread above this ratio
# means the runs are probably not of one point.
LEVEL_JUMP_NOISE_RATIO = 20.0


def level_jump_warnings(axes_list: list[dict[str, np.ndarray]]) -> list[str]:
    """Axes whose mean level differs between runs far more than their noise.

    Each item holds the (packets, samples) arrays of one run. The noise is
    the median over runs of the median per-packet standard deviation.
    """
    if len(axes_list) < 2:
        return []
    messages = []
    for axis, key in AXIS_KEYS:
        means = [float(np.mean(axes[key])) for axes in axes_list]
        noise = float(np.median([
            np.median(np.std(axes[key], axis=1)) for axes in axes_list
        ]))
        spread = max(means) - min(means)
        if noise > 0.0 and spread > LEVEL_JUMP_NOISE_RATIO * noise:
            messages.append(
                f"{axis}: the mean level differs by {spread * 1.0e3:.1f} mg between "
                f"runs ({spread / noise:.0f}x the noise of one sample): the sensor "
                "was probably moved, pool only the runs of one point"
            )
    return messages


def windows_for_groups(groups: list[int]) -> list[np.ndarray]:
    """Support windows that never straddle the pause between two records."""
    total = sum(groups)
    window_packets = max(MINIMUM_WINDOW_PACKETS, total // WINDOW_TARGET_COUNT)
    windows = []
    offset = 0
    for size in groups:
        for index in range(size // window_packets):
            start = offset + index * window_packets
            windows.append(np.arange(start, start + window_packets))
        offset += size
    return windows if len(windows) >= 2 else []


def pooled_name(records: list[LoadedCapture]) -> str:
    """``pooled_141520_143239`` for records named like 20261001_141520_..."""
    parts = []
    for record in records:
        pieces = record.name.split("_")
        parts.append(pieces[1] if len(pieces) > 1 and pieces[1].isdigit() else record.name)
    return "pooled_" + "_".join(parts)


def analyze_capture(
    raw_path: Path,
    settings: SpectrumSettings,
    bands: list[tuple[str, float, float]],
    probe_hz: list[float] | None = None,
    plan: ProbePlan | None = None,
) -> CaptureResult:
    return analyze_records([load_capture(raw_path)], settings, bands, probe_hz, plan)


def analyze_pool(
    raw_paths: list[Path],
    settings: SpectrumSettings,
    bands: list[tuple[str, float, float]],
    probe_hz: list[float] | None = None,
    plan: ProbePlan | None = None,
) -> CaptureResult:
    """Several runs of one point as one long record.

    Each run is cut on its own and the segments of all runs go into one
    average, so the pause between runs never sits inside a segment. Every
    peak of the pooled record also says in how many runs the blind search
    of that run alone finds it, on the same axis within the probe
    tolerance. That count is taken from each run's own search, not from a
    probe at the pooled frequency, which the run itself helped to pick.
    """
    records = [load_capture(path) for path in raw_paths]
    lengths = {record.axes["x"].shape[1] for record in records}
    if len(lengths) > 1:
        raise ValueError(f"runs have different packet lengths {sorted(lengths)}")
    rates = np.array([float(np.mean(record.packet_fs_hz)) for record in records])
    if np.ptp(rates) > MAX_POOL_RATE_MISMATCH * float(np.mean(rates)):
        listed = ", ".join(
            f"{record.name} {rate:.2f} Hz" for record, rate in zip(records, rates)
        )
        raise ValueError(f"runs were recorded at different sampling rates ({listed})")
    result = analyze_records(records, settings, bands, probe_hz, plan)
    tolerance = (plan or ProbePlan()).tolerance_hz
    singles = [analyze_records([record], settings, bands) for record in records]
    peaks = [
        replace(
            peak,
            runs_found=sum(
                any(
                    other.axis == peak.axis
                    and abs(other.frequency_hz - peak.frequency_hz) <= tolerance
                    for other in single.peaks
                )
                for single in singles
            ),
            runs_total=len(records),
        )
        for peak in result.peaks
    ]
    return replace(result, peaks=peaks)


def analyze_records(
    records: list[LoadedCapture],
    settings: SpectrumSettings,
    bands: list[tuple[str, float, float]],
    probe_hz: list[float] | None = None,
    plan: ProbePlan | None = None,
) -> CaptureResult:
    """One record, or several runs pooled, each cut on its own."""
    groups = [record.axes["x"].shape[0] for record in records]
    axes = {
        key: np.vstack([record.axes[key] for record in records])
        for _, key in AXIS_KEYS
    }
    packet_fs_hz = np.concatenate([record.packet_fs_hz for record in records])
    sampling_rate_hz = float(np.mean(packet_fs_hz))
    packet_count, samples_per_packet = axes["x"].shape
    duration_seconds = float(np.sum(samples_per_packet / packet_fs_hz))
    pooled = len(records) > 1

    spectra = [
        analyze_axis(axis, axes[key], sampling_rate_hz, settings, groups)
        for axis, key in AXIS_KEYS
    ]
    bins_tested = sum(spectrum.frequencies.size for spectrum in spectra)
    effective_rows = spectra[0].effective_rows
    z_threshold = significance_threshold(
        bins_tested, settings.alpha, int(round(effective_rows)),
    )
    peaks = find_stable_peaks(spectra, z_threshold, settings, bands)

    windows = windows_for_groups(groups)
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
    if plan is None:
        plan = ProbePlan(requests=tuple(ProbeRequest(float(hz)) for hz in probe_hz or []))
    rows = int(round(effective_rows))
    probe_z, probe_bins = probe_threshold(
        spectra, list(plan.requests), plan.tolerance_hz, plan.alpha, rows,
    )
    probes = probe_frequencies(spectra, list(plan.requests), probe_z, plan.tolerance_hz)
    confirm_skipped = ""
    confirm_z, confirm_bins, confirmations = 0.0, 0, []
    if plan.candidates:
        if any(record.name in plan.candidate_captures for record in records):
            confirm_skipped = (
                "this record is one of those the candidates were found in; "
                "a record cannot confirm its own findings, use another run"
            )
        else:
            confirm_z, confirm_bins = probe_threshold(
                spectra, list(plan.candidates), plan.tolerance_hz, plan.alpha, rows,
            )
            confirmations = probe_frequencies(
                spectra, list(plan.candidates), confirm_z, plan.tolerance_hz,
            )
    # The recording check looks at packets, whatever the segment length,
    # and at each run on its own.
    packet_settings = replace(
        settings,
        nperseg=samples_per_packet,
        noverlap=samples_per_packet // 2,
    )
    member_quality = []
    for record in records:
        if not pooled and settings.nperseg == samples_per_packet:
            packet_spectra = spectra
        else:
            rate = float(np.mean(record.packet_fs_hz))
            packet_spectra = [
                analyze_axis(axis, record.axes[key], rate, packet_settings)
                for axis, key in AXIS_KEYS
            ]
        member_quality.append((
            record.name,
            assess_quality(packet_spectra, record.recorded_fs_hz, record.recorded_axes),
        ))
    return CaptureResult(
        capture=pooled_name(records) if pooled else records[0].name,
        source=records[0].source,
        packet_count=packet_count,
        samples_per_packet=samples_per_packet,
        sampling_rate_hz=sampling_rate_hz,
        duration_seconds=duration_seconds,
        z_threshold=z_threshold,
        bins_tested=bins_tested,
        spectra=spectra,
        peaks=peaks,
        detection_limit_db=detection_limit_db,
        probes=probes,
        quality=member_quality[0][1],
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
        edge_warnings=band_edge_warnings(spectra, z_threshold),
        startup_dropped=any(record.startup_dropped for record in records),
        probe_plan=plan,
        probe_threshold=probe_z,
        probe_bins=probe_bins,
        confirmations=confirmations,
        confirm_threshold=confirm_z,
        confirm_bins=confirm_bins,
        confirm_skipped=confirm_skipped,
        members=tuple(record.name for record in records) if pooled else (),
        pool_warnings=tuple(
            level_jump_warnings([record.axes for record in records])
        ) if pooled else (),
        member_packets=tuple(groups) if pooled else (),
        member_quality=tuple(member_quality) if pooled else (),
    )


# ==========================================================
# Reporting
# ==========================================================


PEAK_FIELDS = [
    "capture", "nperseg", "bin_width_hz", "packet_count", "duration_seconds", "sampling_rate_hz",
    "z_threshold", "axis", "band", "frequency_hz", "bin_frequency_hz",
    "prominence_db",
    "standard_error_db", "z", "half_z_low", "half_z_high", "persistent",
    "support_windows", "support_total", "runs_found", "runs_total",
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
            "bin_frequency_hz": round(peak.bin_frequency_hz, 3),
            "prominence_db": round(peak.prominence_db, 2),
            "standard_error_db": round(peak.standard_error_db, 3),
            "z": round(peak.z, 2),
            "half_z_low": round(peak.half_z_low, 2),
            "half_z_high": round(peak.half_z_high, 2),
            "persistent": int(peak.persistent),
            "support_windows": peak.support_windows,
            "support_total": peak.support_total,
            "runs_found": peak.runs_found,
            "runs_total": peak.runs_total,
        }
        for peak in result.peaks
    ]


PROBE_FIELDS = [
    "capture", "nperseg", "kind", "axis", "requested_hz", "frequency_hz",
    "prominence_db", "standard_error_db", "z", "threshold", "window_bins",
    "tolerance_hz", "alpha", "upper_bound_db", "detected", "source",
]


def probe_rows(result: CaptureResult) -> list[dict[str, Any]]:
    rows = []
    for kind, probes, threshold, bins in (
        ("probe", result.probes, result.probe_threshold, result.probe_bins),
        ("confirm", result.confirmations, result.confirm_threshold, result.confirm_bins),
    ):
        for probe in probes:
            rows.append({
                "capture": result.capture,
                "nperseg": result.nperseg,
                "kind": kind,
                "axis": probe.axis,
                "requested_hz": round(probe.requested_hz, 3),
                "frequency_hz": round(probe.frequency_hz, 3),
                "prominence_db": round(probe.prominence_db, 2),
                "standard_error_db": round(probe.standard_error_db, 3),
                "z": round(probe.z, 2),
                "threshold": round(threshold, 2),
                "window_bins": bins,
                "tolerance_hz": result.probe_plan.tolerance_hz,
                "alpha": result.probe_plan.alpha,
                "upper_bound_db": round(probe.upper_bound_db, 2),
                "detected": int(probe.detected),
                "source": probe.source,
            })
    return rows


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
        f"Significance threshold: z >= {result.z_threshold:.2f}"
        + (
            f"   Each half (Persistent): z >= "
            f"{result.z_threshold * result.spectra[0].half_scale:.2f}"
            if result.spectra else ""
        ),
    ]
    lines[2] += (
        f"   Rate spread: {result.quality.sampling_rate_spread_ppm:.0f} ppm"
    )
    if result.pooled:
        lines[1:2] = [
            f"Pooled record of {len(result.members)} runs, each cut on its own: "
            "no segment crosses the pause between runs",
            *[
                f"  {name}: {packets} packets"
                for name, packets in zip(result.members, result.member_packets)
            ],
            *[f"  ! {message}" for message in result.pool_warnings],
        ]
    lines.append("")
    for name, quality in (result.member_quality or (("", result.quality),)):
        lines.append(f"Recording check{': ' + name if name else ''}:")
        lines.append(
            "  RMS in band: "
            + ", ".join(
                f"{axis} {rms:.2e} g"
                for axis, rms in quality.axis_rms_g.items()
            )
        )
        for message in quality.warnings:
            lines.append(f"  ! {message}")
        if not quality.warnings:
            lines.append("  no anomaly found in the recording itself")
    if result.startup_dropped:
        lines.append(
            "  packet 1 holds a sharp transient (normally the sensor start-up) "
            "and is left out of the spectra"
        )
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
    if result.edge_warnings:
        lines.append("Band edges:")
        for message in result.edge_warnings:
            lines.append(f"  ! {message}")
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
        return "\n".join(lines) + format_probes(result) + format_confirmations(result) + "\n"
    lines.append(f"Significant frequencies: {len(result.peaks)}")
    lines.append(
        "Axis  Band              Freq Hz   Prom dB   SE dB      z   "
        "half z      Persistent" + ("   Runs" if result.pooled else "")
    )
    for peak in result.peaks:
        lines.append(
            f"{peak.axis:<5} {peak.band:<16} {peak.frequency_hz:8.2f}  "
            f"{peak.prominence_db:8.2f}  {peak.standard_error_db:6.3f}  "
            f"{peak.z:5.1f}   "
            f"{peak.half_z_low:4.1f}/{peak.half_z_high:<4.1f}  "
            f"{'yes' if peak.persistent else 'no':>10}"
            + (f"   {peak.runs_label:>4}" if result.pooled else "")
        )
    if result.pooled:
        lines.append(
            "Runs: in how many of the pooled runs the blind search of that run "
            "alone finds the frequency (same axis, within "
            f"{result.probe_plan.tolerance_hz:g} Hz)."
        )
    return "\n".join(lines) + format_probes(result) + format_confirmations(result) + "\n"


def probe_threshold_note(
    plan: ProbePlan, threshold: float, bins: int, full_threshold: float,
) -> str:
    return (
        f"  window +-{plan.tolerance_hz:g} Hz, {bins} bins; z >= {threshold:.2f} "
        f"at alpha {plan.alpha:g} (the blind search of the band needs "
        f"z >= {full_threshold:.2f})"
    )


def probe_lines(probes: list[ProbeResult], present: str, absent: str) -> list[str]:
    lines = ["Axis  Asked   Peak Hz   Prom dB      z   Needed   95% upper   Verdict"]
    for probe in probes:
        verdict = (
            present if probe.detected
            else f"{absent} above {probe.upper_bound_db:.2f} dB"
        )
        lines.append(
            f"{probe.axis:<5} {probe.requested_hz:6.2f}  {probe.frequency_hz:7.2f}  "
            f"{probe.prominence_db:8.2f}  {probe.z:5.1f}  "
            f"{probe.detection_limit_db:7.2f}  {probe.upper_bound_db:10.2f}   "
            f"{verdict}"
        )
    return lines


def format_probes(result: CaptureResult) -> str:
    if not result.probes:
        return ""
    lines = [
        "",
        "Requested frequencies (chosen before this record):",
        probe_threshold_note(
            result.probe_plan, result.probe_threshold, result.probe_bins,
            result.z_threshold,
        ),
        "  The lower threshold is honest only for frequencies taken from",
        "  another run, floor or instrument, not from this record.",
    ]
    lines += probe_lines(result.probes, "present", "absent")
    return "\n" + "\n".join(lines)


def format_confirmations(result: CaptureResult) -> str:
    plan = result.probe_plan
    if not plan.candidates:
        return ""
    lines = [
        "",
        f"Confirmation of {len(plan.candidates)} candidate(s) found by a search on other runs:",
    ]
    lines += [f"  from {source}" for source in plan.candidate_sources]
    if result.confirm_skipped:
        lines.append(f"  skipped: {result.confirm_skipped}")
        return "\n" + "\n".join(lines)
    if not result.confirmations:
        lines.append("  no candidate falls inside the analysed band")
        return "\n" + "\n".join(lines)
    lines.append(probe_threshold_note(
        plan, result.confirm_threshold, result.confirm_bins, result.z_threshold,
    ))
    lines += probe_lines(result.confirmations, "confirmed", "not confirmed,")
    lines.append("Found in:")
    lines += [
        f"  {probe.axis} {probe.requested_hz:.2f} Hz: {probe.source}"
        for probe in result.confirmations
    ]
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
            # On the drawn curve, at the refined frequency.
            height = float(np.interp(
                peak.frequency_hz, spectrum.frequencies, spectrum.mean_psd,
            ))
            panel.plot(
                [peak.frequency_hz], [height],
                marker="x", color="crimson", markersize=9,
                markeredgewidth=2, lw=0, zorder=5,
            )
            support = (
                f"{peak.support_windows}/{peak.support_total} win\n"
                if peak.support_total else ""
            )
            runs = f"  {peak.runs_label}" if peak.runs_label else ""
            panel.annotate(
                f"{peak.frequency_hz:.2f} Hz{runs}\n{support}"
                f"{peak.prominence_db:.2f} dB",
                xy=(peak.frequency_hz, height),
                xytext=(0, 13), textcoords="offset points",
                ha="center", va="bottom", fontsize=8.5, color="crimson",
                zorder=6,
            )
        # Requested or confirmed frequencies that pass the honest threshold
        # but not the blind one: shown, but told apart from the peaks.
        marked = {round(peak.frequency_hz, 2) for peak in axis_peaks}
        honest = [
            (probe, "conf." if probe in result.confirmations else "asked")
            for probe in [*result.probes, *result.confirmations]
            if probe.axis == spectrum.axis and probe.detected
            and round(probe.frequency_hz, 2) not in marked
        ]
        for probe, kind in honest:
            height = float(np.interp(
                probe.frequency_hz, spectrum.frequencies, spectrum.mean_psd,
            ))
            panel.plot(
                [probe.frequency_hz], [height],
                marker="o", markerfacecolor="none", markeredgecolor="darkgreen",
                markersize=10, markeredgewidth=1.8, lw=0, zorder=5,
            )
            panel.annotate(
                f"{probe.frequency_hz:.2f} Hz\n{kind} z={probe.z:.1f}",
                xy=(probe.frequency_hz, height),
                xytext=(0, 13), textcoords="offset points",
                ha="center", va="bottom", fontsize=8.5, color="darkgreen",
                zorder=6,
            )
        top = max(
            float(np.max(spectrum.mean_psd)), float(np.max(ceiling)),
        )
        panel.set_ylim(0.0, top * (1.42 if axis_peaks or honest else 1.08))
        panel.set_ylabel(f"{spectrum.axis} axis\nPSD [g$^2$/Hz]")
        panel.grid(True, alpha=0.3)
        if not axis_peaks and not honest:
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
            Line2D([0], [0], marker="o", markerfacecolor="none",
                   markeredgecolor="darkgreen", markersize=9,
                   markeredgewidth=1.8, lw=0),
        ],
        labels=[
            "sensor noise: nothing here is a structure",
            "detection threshold",
            "broadband floor",
            "measured PSD (axis colour)",
            "frequency that stands out",
            "requested / confirmed, honest threshold",
        ],
        loc="lower center", ncol=3, fontsize=9,
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
    figure.tight_layout(rect=(0, 0.06, 1, 0.925))
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
            on_curve_db = float(np.interp(
                peak.frequency_hz, spectrum.frequencies, spectrum.prominence_db,
            ))
            panel.plot(
                [peak.frequency_hz], [on_curve_db],
                marker="v", color="crimson", markersize=8, lw=0,
            )
            runs = f"  {peak.runs_label}" if peak.runs_label else ""
            panel.annotate(
                f"{peak.frequency_hz:.2f} Hz{runs}\nz={peak.z:.1f}"
                + ("" if peak.persistent else "\n(one half only)"),
                xy=(peak.frequency_hz, on_curve_db),
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


def expand_raw_paths(paths: list[Path]) -> list[Path]:
    """Files as given; a folder is one measuring point, all its runs.

    Only ``*_raw.npz`` directly in the folder count, in name order:
    subfolders are other points and are never mixed in.
    """
    expanded: list[Path] = []
    for path in paths:
        if not path.is_dir():
            expanded.append(path)
            continue
        runs = sorted(path.glob("*_raw.npz"))
        if not runs:
            inner = sorted(
                item.name for item in path.iterdir()
                if item.is_dir() and any(item.glob("*_raw.npz"))
            )
            hint = f"; point folders inside: {', '.join(inner)}" if inner else ""
            raise ValueError(f"no *_raw.npz in {path}{hint}")
        expanded.extend(runs)
    return list(dict.fromkeys(expanded))


def resolve_output_directory(
    output: Path | None, raw_paths: list[Path], pool: bool = False,
) -> Path:
    """``raw_paths`` as given on the command line, folders not expanded."""
    if output is not None:
        return output
    if len(raw_paths) == 1 and raw_paths[0].is_dir():
        # One directory per point, named as overview_figure.py names it:
        # the runs one by one in runs/, pooled in pool/ next to it.
        point = STABLE_RESULTS_DIRECTORY / raw_paths[0].resolve().name
        return point / ("pool" if pool else "runs")
    if len(raw_paths) == 1:
        return STABLE_RESULTS_DIRECTORY / raw_paths[0].stem
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return STABLE_RESULTS_DIRECTORY / f"{'pooled' if pool else 'multi'}_{stamp}"


def run_single_resolution(
    raw_paths: list[Path],
    output_directory: Path,
    settings: SpectrumSettings,
    probe_hz: list[float] | None = None,
    plan: ProbePlan | None = None,
    pool: bool = False,
) -> list[CaptureResult]:
    """Analyse every capture at one segment length into one directory."""
    bands = load_band_names()
    output_directory.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    all_probe_rows: list[dict[str, Any]] = []
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
    jobs = [raw_paths] if pool else [[raw_path] for raw_path in raw_paths]
    for job in jobs:
        try:
            if pool:
                result = analyze_pool(job, settings, bands, probe_hz, plan)
            else:
                result = analyze_capture(job[0], settings, bands, probe_hz, plan)
        except ValueError as error:
            label = "pooled runs" if pool else job[0].stem
            message = (
                f"{label}: skipped at nperseg {settings.nperseg}: "
                f"{error}"
            )
            print(message)
            report_parts.append("=" * 72)
            report_parts.append(message + "\n")
            continue
        results.append(result)
        all_rows.extend(peak_rows(result))
        all_probe_rows.extend(probe_rows(result))
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
        for message in result.pool_warnings:
            print(f"WARNING: {message}")
        print(
            f"{result.capture} [nperseg {result.nperseg}]: "
            f"{result.row_count} rows, z>={result.z_threshold:.2f} -> "
            f"{peak_summary(result)}"
            + (
                f"  [band edge: {len(result.edge_warnings)} warning(s), "
                "see report]"
                if result.edge_warnings else ""
            )
        )
    write_csv_rows(output_directory / "stable_frequencies.csv", all_rows)
    if all_probe_rows:
        with (output_directory / "stable_probes.csv").open(
            "w", encoding="utf-8", newline="",
        ) as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=PROBE_FIELDS)
            writer.writeheader()
            writer.writerows(all_probe_rows)
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
                height = float(np.interp(
                    peak.frequency_hz, spectrum.frequencies, spectrum.mean_psd,
                ))
                panel.plot(
                    [peak.frequency_hz],
                    [np.sqrt(height) * 1.0e6],
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
    plan: ProbePlan | None = None,
    pool: bool = False,
) -> Path:
    """Analyse captures, once per segment length.

    Without ``nperseg_values`` the configured segment length is written
    straight into ``output_directory``. With them, each length gets its own
    ``nperseg_<n>/`` directory, and ``comparison.txt`` plus one overlay
    figure per capture are written next to them.
    """
    if not nperseg_values:
        run_single_resolution(raw_paths, output_directory, settings, probe_hz, plan, pool)
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
            plan,
            pool,
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
    parser.add_argument(
        "raw_paths", nargs="+", type=Path, metavar="PATH",
        help="*_raw.npz files, or a folder of one measuring point: all "
             "*_raw.npz directly in it are its runs",
    )
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
        "--nperseg", type=int, nargs="+", default=None, metavar="SAMPLES",
        help="segment lengths to analyse, each into its own nperseg_<n> "
             "directory (default: [stable_spectrum] nperseg in config.toml, "
             "2048; give 1024 2048 4096 to compare resolutions)",
    )
    parser.add_argument(
        "--probe", type=parse_probe, nargs="+", default=None, metavar="HZ",
        help="frequencies chosen before this record (2.88 on every axis, "
             "Y:2.88 on one): what sits there, judged with a threshold that "
             "counts only the bins near them, and the 95%% upper bound when "
             "nothing is found",
    )
    parser.add_argument(
        "--probe-tolerance", type=float, default=DEFAULT_PROBE_TOLERANCE_HZ,
        metavar="HZ",
        help="how far from a requested frequency its peak may sit "
             f"(default {DEFAULT_PROBE_TOLERANCE_HZ:g} Hz)",
    )
    parser.add_argument(
        "--probe-alpha", type=float, default=DEFAULT_ALPHA,
        help="false-positive rate for --probe and --confirm-from, independent "
             f"of --alpha (default {DEFAULT_ALPHA:g})",
    )
    parser.add_argument(
        "--pool", action="store_true",
        help="treat the given records as runs of one point and analyse them as "
             "one long record: each run is cut on its own and all segments go "
             "into one average; every peak says in how many runs it is found "
             "on its own (e.g. 1/2)",
    )
    parser.add_argument(
        "--confirm-from", type=Path, nargs="+", default=None, metavar="PATH",
        help="stable_frequencies.csv files, or folders holding them, from a "
             "search on other runs (e.g. --alpha 0.05): their peaks are "
             "checked on these records as frequencies chosen in advance",
    )
    parser.add_argument(
        "--baseline-window", type=float, default=None, metavar="HZ",
        help="width of the running-median baseline (default: from "
             f"config.toml, {DEFAULT_BASELINE_WINDOW_HZ:g} Hz without it)",
    )
    parser.add_argument(
        "--min-distance", type=float, default=None, metavar="HZ",
        help="peaks closer than this merge (default: [stable_spectrum] "
             "min_distance_hz, else the bands of config.toml, 1 Hz)",
    )
    parser.add_argument(
        "--separation-sigma", type=float, default=None, metavar="K",
        help="keep a weaker peak within --min-distance of a stronger one when "
             "the dip between them is deeper than K standard errors "
             "(default: [stable_spectrum] separation_sigma, else "
             f"{DEFAULT_SEPARATION_SIGMA:g}; 0 merges every close pair)",
    )
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    cli = parse_cli_arguments(arguments)
    band = tuple(cli.band) if cli.band is not None else None
    settings = load_settings(
        cli.alpha, band, cli.baseline_window, cli.min_distance, cli.separation_sigma,
    )
    candidates: tuple[ProbeRequest, ...] = ()
    candidate_captures: frozenset[str] = frozenset()
    candidate_sources: tuple[str, ...] = ()
    if cli.confirm_from:
        try:
            candidates, candidate_captures, candidate_sources = load_candidates(
                cli.confirm_from, cli.probe_tolerance,
            )
        except (OSError, ValueError, KeyError) as error:
            print(f"error: --confirm-from: {error}")
            return 1
    plan = ProbePlan(
        requests=tuple(cli.probe or ()),
        candidates=candidates,
        candidate_captures=candidate_captures,
        candidate_sources=candidate_sources,
        tolerance_hz=cli.probe_tolerance,
        alpha=cli.probe_alpha,
    )
    try:
        raw_paths = expand_raw_paths(cli.raw_paths)
    except (OSError, ValueError) as error:
        print(f"error: {error}")
        return 1
    run_stable_spectrum(
        raw_paths,
        resolve_output_directory(cli.output, cli.raw_paths, cli.pool),
        settings,
        None,
        cli.nperseg or load_default_nperseg(),
        plan,
        cli.pool,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
