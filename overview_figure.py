"""Overview figure for a folder of raw captures.

Joins every ``*.npz`` capture in a folder and draws two panels:

1. the measured spectrum of each capture from 0 Hz to Nyquist, all
   captures overlaid, one colour per axis, with the frequencies that stand
   out of the pooled record labelled. Structures outside the analysis band
   (machinery, mains-related lines) show up here.
2. the analysis band (from ``config.toml``, 0.5-15 Hz by default) for all
   captures pooled: prominence over the local baseline against the band
   that the pooled record cannot tell apart from sensor noise.

Captures are pooled packet by packet, so they must share the sampling rate
and the packet length. The statistics are those of ``stable_spectrum.py``
at one periodogram per packet.

Usage::

    python overview_figure.py "C:/path/to/folder with npz"
    python overview_figure.py results --output stable_results/overview.png
    python overview_figure.py results --band 0.5 30
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from stable_spectrum import (
    AXIS_COLORS,
    AXIS_KEYS,
    STABLE_RESULTS_DIRECTORY,
    AxisSpectrum,
    StablePeak,
    analyze_axis,
    find_stable_peaks,
    load_band_names,
    load_settings,
    significance_threshold,
)

# Captures whose mean sampling rates differ by more than this cannot share
# one frequency grid without smearing lines.
MAX_RATE_MISMATCH = 0.005
# The first bins sit on the DC removal and the last on the anti-alias
# roll-off; neither says anything about the structure.
FULL_BAND_START_HZ = 0.5
FULL_BAND_NYQUIST_MARGIN = 0.02
MAX_LABELS = 8
LABEL_MERGE_HZ = 1.0


@dataclass(frozen=True)
class Capture:
    name: str
    axes: dict[str, np.ndarray]
    sampling_rate_hz: float


@dataclass(frozen=True)
class Overview:
    captures: list[Capture]
    sampling_rate_hz: float
    full_band: dict[str, list[AxisSpectrum]]
    full_band_peaks: list[StablePeak]
    pooled: list[AxisSpectrum]
    pooled_peaks: list[StablePeak]
    pooled_threshold: float
    band_hz: tuple[float, float]


def load_folder(folder: Path) -> list[Capture]:
    captures = []
    for path in sorted(folder.glob("*.npz")):
        try:
            archive = np.load(path, allow_pickle=False)
        except (OSError, ValueError, zipfile.BadZipFile) as error:
            print(f"skipped {path.name}: not readable ({error})")
            continue
        with archive:
            missing = [key for key in ("x", "y", "z", "packet_fs_hz")
                       if key not in archive.files]
            if missing:
                print(f"skipped {path.name}: no {', '.join(missing)}")
                continue
            captures.append(Capture(
                name=path.stem,
                axes={key: np.asarray(archive[key]) for _, key in AXIS_KEYS},
                sampling_rate_hz=float(np.mean(archive["packet_fs_hz"])),
            ))
    if not captures:
        raise ValueError(f"no raw captures (*.npz) in {folder}")
    lengths = {capture.axes["x"].shape[1] for capture in captures}
    if len(lengths) > 1:
        raise ValueError(
            f"captures have different packet lengths {sorted(lengths)}; "
            "put only compatible captures in one folder"
        )
    rates = np.array([capture.sampling_rate_hz for capture in captures])
    if np.ptp(rates) > MAX_RATE_MISMATCH * float(np.mean(rates)):
        listed = ", ".join(
            f"{capture.name} {capture.sampling_rate_hz:.2f} Hz"
            for capture in captures
        )
        raise ValueError(
            f"captures were recorded at different sampling rates ({listed}); "
            "put only one ODR in one folder"
        )
    return captures


def peaks_in(
    spectra: list[AxisSpectrum],
    packet_count: int,
    settings,
) -> tuple[list[StablePeak], float]:
    bins_tested = sum(spectrum.frequencies.size for spectrum in spectra)
    threshold = significance_threshold(
        bins_tested, settings.alpha, packet_count,
    )
    peaks = find_stable_peaks(spectra, threshold, settings, load_band_names())
    return peaks, threshold


def build_overview(
    captures: list[Capture],
    band_hz: tuple[float, float] | None = None,
    alpha: float = 0.01,
) -> Overview:
    sampling_rate_hz = float(np.mean(
        [capture.sampling_rate_hz for capture in captures]
    ))
    settings = load_settings(alpha, band_hz)
    samples_per_packet = captures[0].axes["x"].shape[1]
    settings = replace(
        settings,
        nperseg=samples_per_packet,
        noverlap=samples_per_packet // 2,
    )
    nyquist = sampling_rate_hz / 2.0
    full_settings = replace(
        settings,
        band_hz=(
            FULL_BAND_START_HZ,
            nyquist * (1.0 - FULL_BAND_NYQUIST_MARGIN),
        ),
    )
    full_band = {
        axis: [
            analyze_axis(axis, capture.axes[key], sampling_rate_hz, full_settings)
            for capture in captures
        ]
        for axis, key in AXIS_KEYS
    }
    stacked = {
        key: np.vstack([capture.axes[key] for capture in captures])
        for _, key in AXIS_KEYS
    }
    packet_count = stacked["x"].shape[0]
    full_pooled = [
        analyze_axis(axis, stacked[key], sampling_rate_hz, full_settings)
        for axis, key in AXIS_KEYS
    ]
    full_band_peaks, _ = peaks_in(full_pooled, packet_count, full_settings)
    pooled = [
        analyze_axis(axis, stacked[key], sampling_rate_hz, settings)
        for axis, key in AXIS_KEYS
    ]
    pooled_peaks, pooled_threshold = peaks_in(pooled, packet_count, settings)
    return Overview(
        captures=captures,
        sampling_rate_hz=sampling_rate_hz,
        full_band=full_band,
        full_band_peaks=full_band_peaks,
        pooled=pooled,
        pooled_peaks=pooled_peaks,
        pooled_threshold=pooled_threshold,
        band_hz=settings.band_hz,
    )


def label_groups(peaks: list[StablePeak]) -> list[list[StablePeak]]:
    """Peaks of different axes at the same frequency share one label.

    Keeps the strongest groups first and at most MAX_LABELS in total.
    """
    groups: list[list[StablePeak]] = []
    for peak in sorted(peaks, key=lambda item: -item.z):
        for group in groups:
            if abs(group[0].frequency_hz - peak.frequency_hz) <= LABEL_MERGE_HZ:
                group.append(peak)
                break
        else:
            groups.append([peak])
    return groups[:MAX_LABELS]


def save_overview(path: Path, overview: Overview) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    count = len(overview.captures)
    packets = sum(capture.axes["x"].shape[0] for capture in overview.captures)
    samples_per_packet = overview.captures[0].axes["x"].shape[1]
    minutes = packets * samples_per_packet / overview.sampling_rate_hz / 60.0
    low, high = overview.band_hz

    figure, (spectrum_panel, band_panel) = plt.subplots(
        2, 1, figsize=(12, 9.5),
    )

    # Panel 1: every capture, full band.
    alphas = np.linspace(0.45, 0.95, count) if count > 1 else [0.9]
    top = 0.0
    bottom = np.inf
    for axis, _ in AXIS_KEYS:
        for alpha, spectrum in zip(alphas, overview.full_band[axis]):
            density = np.sqrt(spectrum.mean_psd) * 1.0e6
            spectrum_panel.semilogy(
                spectrum.frequencies, density,
                color=AXIS_COLORS[axis], lw=0.8, alpha=float(alpha),
            )
            top = max(top, float(np.max(density)))
            bottom = min(bottom, float(np.min(density)))
    spectrum_panel.axvspan(low, high, color="grey", alpha=0.12)
    for group in label_groups(overview.full_band_peaks):
        axes_in_group = sorted({peak.axis for peak in group})
        height = 0.0
        for peak in group:
            spectra = overview.full_band[peak.axis]
            index = int(np.argmin(
                np.abs(spectra[0].frequencies - peak.frequency_hz)
            ))
            height = max(height, *(
                float(np.sqrt(spectrum.mean_psd[index])) * 1.0e6
                for spectrum in spectra
            ))
        color = (
            AXIS_COLORS[axes_in_group[0]] if len(axes_in_group) == 1 else "0.15"
        )
        strongest = max(group, key=lambda peak: peak.prominence_db)
        spectrum_panel.annotate(
            f"{', '.join(axes_in_group)} {strongest.frequency_hz:.1f} Гц\n"
            f"+{strongest.prominence_db:.1f} дБ",
            xy=(strongest.frequency_hz, height),
            xytext=(0, 14), textcoords="offset points",
            ha="center", va="bottom", fontsize=8, color=color,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
            arrowprops=dict(arrowstyle="-", color=color, lw=0.6),
        )
    spectrum_panel.set_xlim(0.0, overview.sampling_rate_hz / 2.0)
    spectrum_panel.set_ylim(bottom * 0.7, top * 2.2)
    spectrum_panel.set_xlabel("Гц")
    spectrum_panel.set_ylabel("µg/√Гц")
    spectrum_panel.grid(True, which="both", alpha=0.25)
    spectrum_panel.set_title(
        f"Спектр 0–{overview.sampling_rate_hz / 2.0:.0f} Гц, записей: {count} "
        "(по каждой оси — по кривой на запись)",
        fontsize=11,
    )
    spectrum_panel.legend(
        handles=[
            *[Line2D([0], [0], color=AXIS_COLORS[axis], lw=1.2)
              for axis, _ in AXIS_KEYS],
            Patch(color="grey", alpha=0.12),
        ],
        labels=[
            *[axis for axis, _ in AXIS_KEYS],
            f"полоса анализа {low:g}–{high:g} Гц",
        ],
        loc="upper right", fontsize=8,
    )

    # Panel 2: analysis band, all captures pooled.
    extent = 0.0
    for spectrum in overview.pooled:
        color = AXIS_COLORS[spectrum.axis]
        limit = overview.pooled_threshold * spectrum.standard_error_db
        band_panel.fill_between(
            spectrum.frequencies, -limit, limit, color=color, alpha=0.08, lw=0,
        )
        band_panel.plot(
            spectrum.frequencies, spectrum.prominence_db,
            color=color, lw=1.2, label=f"{spectrum.axis} ({packets} пакетов)",
        )
        extent = max(
            extent,
            float(np.max(np.abs(spectrum.prominence_db))),
            float(np.max(limit)),
        )
    for peak in overview.pooled_peaks:
        band_panel.plot(
            [peak.frequency_hz], [peak.prominence_db],
            marker="x", color="crimson", markersize=9, markeredgewidth=2, lw=0,
        )
        band_panel.annotate(
            f"{peak.axis} {peak.frequency_hz:.2f} Гц\nz={peak.z:.1f}",
            xy=(peak.frequency_hz, peak.prominence_db),
            xytext=(0, 10), textcoords="offset points",
            ha="center", fontsize=8, color="crimson",
        )
    band_panel.axhline(0.0, color="k", lw=0.5)
    band_panel.set_xlim(low, high)
    band_panel.set_ylim(-1.3 * extent, 1.6 * extent)
    band_panel.set_xlabel("Гц")
    band_panel.set_ylabel("превышение над базой, дБ")
    band_panel.grid(True, alpha=0.25)
    band_panel.legend(loc="upper right", fontsize=8)
    if overview.pooled_peaks:
        verdict = "выходят: " + ", ".join(
            f"{peak.axis} {peak.frequency_hz:.2f} Гц"
            for peak in sorted(overview.pooled_peaks, key=lambda p: p.frequency_hz)
        )
    else:
        verdict = "ни одна кривая не выходит"
    band_panel.set_title(
        f"{low:g}–{high:g} Гц: все записи вместе, {minutes:.0f} мин; "
        "закрашено — неотличимо от шума "
        f"(порог z={overview.pooled_threshold:.1f}); {verdict}",
        fontsize=11,
    )

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=130)
    plt.close(figure)


def parse_cli_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay the spectra of all raw captures in a folder",
    )
    parser.add_argument("folder", type=Path, help="folder with *.npz captures")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="PNG path (default: stable_results/<folder>/figure_overview.png)",
    )
    parser.add_argument(
        "--band", type=float, nargs=2, default=None,
        metavar=("MIN_HZ", "MAX_HZ"),
        help="analysis band of the second panel (default: from config.toml)",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.01,
        help="family-wise false-positive rate across all bins and axes",
    )
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    cli = parse_cli_arguments(arguments)
    try:
        captures = load_folder(cli.folder)
    except ValueError as error:
        print(f"error: {error}")
        return 1
    overview = build_overview(
        captures,
        tuple(cli.band) if cli.band is not None else None,
        cli.alpha,
    )
    output = cli.output or (
        STABLE_RESULTS_DIRECTORY / cli.folder.resolve().name / "figure_overview.png"
    )
    save_overview(output, overview)
    for capture in captures:
        print(f"  {capture.name}: {capture.axes['x'].shape[0]} packets")
    listed = ", ".join(
        f"{peak.axis} {peak.frequency_hz:.1f} Hz" for peak in overview.full_band_peaks
    ) or "none"
    print(f"Full band, pooled: {listed}")
    listed = ", ".join(
        f"{peak.axis} {peak.frequency_hz:.2f} Hz" for peak in overview.pooled_peaks
    ) or "none"
    print(
        f"{overview.band_hz[0]:g}-{overview.band_hz[1]:g} Hz, pooled "
        f"(z >= {overview.pooled_threshold:.2f}): {listed}"
    )
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
