"""Overview figure for a folder of raw captures.

Joins every ``*.npz`` capture in a folder and draws two panels:

1. the measured spectrum of each capture from 0 Hz to Nyquist, all
   captures overlaid, one colour per axis, with the frequencies that stand
   out of the pooled record labelled. Structures outside the analysis band
   (machinery, mains-related lines) show up here.
2. the analysis band (from ``config.toml``, 0.2-15 Hz by default) for all
   captures pooled: prominence over the local baseline against the band
   that the pooled record cannot tell apart from sensor noise.

Captures are pooled packet by packet, so they must share the sampling rate
and the packet length. The statistics are those of ``stable_spectrum.py``.

One figure is drawn per segment length (``--nperseg``, 1024 2048 4096 by
default), each into its own ``nperseg_<n>/`` directory as
``stable_spectrum.py`` lays them out. A segment longer than a packet joins
neighbouring packets; the packets of all captures are joined in one row,
so at every join between two captures one segment holds the end of one
capture and the start of the next. On 16.09 the level steps at those joins
were no larger than the steps between packets inside a capture, so that
segment is an ordinary one as long as the sensor stayed in place.

The default alpha 0.01 gives the result. A looser alpha (``--alpha 0.05``)
is a search mode: the shading stays at the 0.01 threshold, and the peaks
that pass only the looser one are drawn hollow and listed apart.

Besides the peaks, the console lists the dips: the same search run on the
profile turned upside down. They are not a false-peak count: on 02.10.2026
narrow dips turned up on the records with strong building lines and
repeated at one frequency across resolutions, while the quiet days had
none.

Usage::

    python overview_figure.py "C:/path/to/folder with npz"
    python overview_figure.py results --output stable_results/overview
    python overview_figure.py results --band 0.5 30 --nperseg 1024
    python overview_figure.py results --alpha 0.05
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from scipy.signal import find_peaks

from stable_spectrum import (
    AXIS_COLORS,
    AXIS_KEYS,
    DEFAULT_ALPHA,
    DEFAULT_NPERSEG_SWEEP,
    MINIMUM_RELIABLE_PACKETS,
    STABLE_RESULTS_DIRECTORY,
    AxisSpectrum,
    StablePeak,
    analyze_axis,
    drop_startup_packet,
    find_stable_peaks,
    load_band_names,
    load_settings,
    resolution_directory_name,
    significance_threshold,
)

# Captures whose mean sampling rates differ by more than this cannot share
# one frequency grid without smearing lines.
MAX_RATE_MISMATCH = 0.005
# The full-band panel starts where the analysis band of config.toml starts,
# so both panels and every other script share one lower edge. The last bins
# sit on the anti-alias roll-off and are left out.
FULL_BAND_NYQUIST_MARGIN = 0.02
MAX_LABELS = 8
LABEL_MERGE_HZ = 1.0
# Hann leakage spreads a noise excursion over two or three neighbouring
# bins; a wider dip is a trough of the baseline beside a broad structure
# rather than a noise excursion.
NARROW_DIP_MAX_BINS = 3


@dataclass(frozen=True)
class Capture:
    name: str
    axes: dict[str, np.ndarray]
    sampling_rate_hz: float


@dataclass(frozen=True)
class Dip:
    axis: str
    frequency_hz: float
    z: float
    width_bins: int

    @property
    def narrow(self) -> bool:
        return self.width_bins <= NARROW_DIP_MAX_BINS


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
    nperseg: int
    alpha: float
    # The threshold at the default alpha, so that peaks passing only a
    # looser alpha can be told apart.
    reference_threshold: float
    pooled_dips: list[Dip]

    @property
    def bin_width_hz(self) -> float:
        frequencies = self.pooled[0].frequencies
        return float(frequencies[1] - frequencies[0])

    @property
    def segment_count(self) -> int:
        return self.pooled[0].row_count

    @property
    def effective_segments(self) -> float:
        return self.pooled[0].effective_rows

    @property
    def searching(self) -> bool:
        """A looser alpha than the default: a search, not a result."""
        return self.pooled_threshold < self.reference_threshold

    def passes_reference(self, peak: StablePeak) -> bool:
        return peak.z >= self.reference_threshold


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
            axes, packet_fs_hz, _ = drop_startup_packet(
                {key: np.asarray(archive[key]) for _, key in AXIS_KEYS},
                np.asarray(archive["packet_fs_hz"]),
            )
            captures.append(Capture(
                name=path.stem,
                axes=axes,
                sampling_rate_hz=float(np.mean(packet_fs_hz)),
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


def threshold_for(spectra: list[AxisSpectrum], alpha: float) -> float:
    """The threshold of ``stable_spectrum.py``: overlapping segments count
    as the number of independent ones they are worth."""
    bins_tested = sum(spectrum.frequencies.size for spectrum in spectra)
    return significance_threshold(
        bins_tested, alpha, int(round(spectra[0].effective_rows)),
    )


def peaks_in(
    spectra: list[AxisSpectrum],
    settings,
) -> tuple[list[StablePeak], float]:
    threshold = threshold_for(spectra, settings.alpha)
    peaks = find_stable_peaks(spectra, threshold, settings, load_band_names())
    return peaks, threshold


def find_dips(
    spectra: list[AxisSpectrum],
    threshold: float,
    settings,
) -> list[Dip]:
    """The peak search run on the profile turned upside down.

    A structure of its own, not a false-peak count: a building does give
    narrow dips, repeating at one frequency across resolutions.
    """
    dips = []
    for spectrum in spectra:
        bin_width_hz = float(spectrum.frequencies[1] - spectrum.frequencies[0])
        distance = max(1, int(round(settings.min_distance_hz / bin_width_hz)))
        if spectrum.context_z is not None:
            search_z, offset = spectrum.context_z, spectrum.band_offset
        else:
            search_z, offset = spectrum.z, 0
        found, _ = find_peaks(-search_z, height=threshold, distance=distance)
        below = search_z < -threshold
        for index in found:
            band_index = int(index) - offset
            if not 0 <= band_index < spectrum.z.size:
                continue
            start = int(index)
            while start > 0 and below[start - 1]:
                start -= 1
            stop = int(index)
            while stop + 1 < below.size and below[stop + 1]:
                stop += 1
            dips.append(Dip(
                axis=spectrum.axis,
                frequency_hz=float(spectrum.frequencies[band_index]),
                z=float(spectrum.z[band_index]),
                width_bins=stop - start + 1,
            ))
    dips.sort(key=lambda dip: dip.z)
    return dips


def build_overview(
    captures: list[Capture],
    band_hz: tuple[float, float] | None = None,
    alpha: float = DEFAULT_ALPHA,
    baseline_window_hz: float | None = None,
    nperseg: int | None = None,
) -> Overview:
    """Both panels at one segment length, one packet long by default.

    Raises ValueError when the captures are too short for the segment.
    """
    sampling_rate_hz = float(np.mean(
        [capture.sampling_rate_hz for capture in captures]
    ))
    settings = load_settings(alpha, band_hz, baseline_window_hz)
    segment = nperseg or captures[0].axes["x"].shape[1]
    settings = replace(settings, nperseg=segment, noverlap=segment // 2)
    nyquist = sampling_rate_hz / 2.0
    configured_start_hz = load_settings(alpha, None).band_hz[0]
    full_settings = replace(
        settings,
        band_hz=(
            configured_start_hz,
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
    full_pooled = [
        analyze_axis(axis, stacked[key], sampling_rate_hz, full_settings)
        for axis, key in AXIS_KEYS
    ]
    full_band_peaks, _ = peaks_in(full_pooled, full_settings)
    pooled = [
        analyze_axis(axis, stacked[key], sampling_rate_hz, settings)
        for axis, key in AXIS_KEYS
    ]
    pooled_peaks, pooled_threshold = peaks_in(pooled, settings)
    return Overview(
        captures=captures,
        sampling_rate_hz=sampling_rate_hz,
        full_band=full_band,
        full_band_peaks=full_band_peaks,
        pooled=pooled,
        pooled_peaks=pooled_peaks,
        pooled_threshold=pooled_threshold,
        band_hz=settings.band_hz,
        nperseg=segment,
        alpha=alpha,
        reference_threshold=threshold_for(pooled, DEFAULT_ALPHA),
        pooled_dips=find_dips(pooled, pooled_threshold, settings),
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
    resolution = (
        f"nperseg {overview.nperseg}, бин {overview.bin_width_hz:.3f} Гц"
    )
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
        f"(по каждой оси — по кривой на запись); {resolution}",
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
    if overview.nperseg > samples_per_packet:
        rows_label = (
            f"{overview.segment_count} отрезков, независимых "
            f"{overview.effective_segments:.0f}"
        )
    else:
        rows_label = f"{packets} пакетов"
    # The shading stays at the standard threshold; in search mode the peaks
    # that pass only the looser one are drawn hollow inside it.
    searching = overview.searching
    shade_threshold = max(overview.pooled_threshold, overview.reference_threshold)
    extent = 0.0
    for spectrum in overview.pooled:
        color = AXIS_COLORS[spectrum.axis]
        limit = shade_threshold * spectrum.standard_error_db
        band_panel.fill_between(
            spectrum.frequencies, -limit, limit, color=color, alpha=0.08, lw=0,
        )
        band_panel.plot(
            spectrum.frequencies, spectrum.prominence_db,
            color=color, lw=1.2, label=f"{spectrum.axis} ({rows_label})",
        )
        extent = max(
            extent,
            float(np.max(np.abs(spectrum.prominence_db))),
            float(np.max(limit)),
        )
    pooled_by_axis = {spectrum.axis: spectrum for spectrum in overview.pooled}
    for peak in overview.pooled_peaks:
        pooled = pooled_by_axis[peak.axis]
        on_curve_db = float(np.interp(
            peak.frequency_hz, pooled.frequencies, pooled.prominence_db,
        ))
        if overview.passes_reference(peak):
            band_panel.plot(
                [peak.frequency_hz], [on_curve_db],
                marker="x", color="crimson", markersize=9, markeredgewidth=2, lw=0,
            )
        else:
            band_panel.plot(
                [peak.frequency_hz], [on_curve_db],
                marker="o", markerfacecolor="none", markeredgecolor="crimson",
                markersize=9, markeredgewidth=1.5, lw=0,
            )
        band_panel.annotate(
            f"{peak.axis} {peak.frequency_hz:.2f} Гц\nz={peak.z:.1f}",
            xy=(peak.frequency_hz, on_curve_db),
            xytext=(0, 10), textcoords="offset points",
            ha="center", fontsize=8, color="crimson",
        )
    band_panel.axhline(0.0, color="k", lw=0.5)
    band_panel.set_xlim(low, high)
    band_panel.set_ylim(-1.3 * extent, 1.6 * extent)
    band_panel.set_xlabel("Гц")
    band_panel.set_ylabel("превышение над базой, дБ")
    band_panel.grid(True, alpha=0.25)
    handles, labels = band_panel.get_legend_handles_labels()
    if searching:
        handles.append(Line2D(
            [0], [0], marker="o", markerfacecolor="none",
            markeredgecolor="crimson", markeredgewidth=1.5, lw=0,
        ))
        labels.append(
            f"только поиск: z от {overview.pooled_threshold:.2f} "
            f"до {overview.reference_threshold:.2f}"
        )
    band_panel.legend(handles, labels, loc="upper right", fontsize=8)
    ordered = sorted(overview.pooled_peaks, key=lambda p: p.frequency_hz)
    confirmed = [peak for peak in ordered if overview.passes_reference(peak)]
    searched = [peak for peak in ordered if not overview.passes_reference(peak)]
    if confirmed:
        verdict = "выходят: " + ", ".join(
            f"{peak.axis} {peak.frequency_hz:.2f} Гц" for peak in confirmed
        )
    else:
        verdict = "ни одна кривая не выходит"
    if searched:
        verdict += "; только поиск: " + ", ".join(
            f"{peak.axis} {peak.frequency_hz:.2f} Гц" for peak in searched
        )
    caution = ""
    if overview.effective_segments < MINIMUM_RELIABLE_PACKETS:
        caution = (
            f"\nвнимание: независимых отрезков "
            f"{overview.effective_segments:.0f}, меньше "
            f"{MINIMUM_RELIABLE_PACKETS}, погрешность сама шумит"
        )
    shade_alpha = DEFAULT_ALPHA if searching else overview.alpha
    search_note = (
        f"; режим поиска alpha {overview.alpha:g}, z={overview.pooled_threshold:.2f}"
        if searching else ""
    )
    band_panel.set_title(
        f"{low:g}–{high:g} Гц: все записи вместе, {minutes:.0f} мин, "
        f"{resolution}\nзакрашено — неотличимо от шума "
        f"(порог z={shade_threshold:.2f} при alpha {shade_alpha:g}"
        f"{search_note})\n{verdict}{caution}",
        fontsize=10,
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
        help="output directory; each segment length goes into its own "
             "nperseg_<n>/ (default: stable_results/<folder>)",
    )
    parser.add_argument(
        "--nperseg", type=int, nargs="+",
        default=list(DEFAULT_NPERSEG_SWEEP), metavar="SAMPLES",
        help="segment lengths, one figure each (default: 1024 2048 4096)",
    )
    parser.add_argument(
        "--band", type=float, nargs=2, default=None,
        metavar=("MIN_HZ", "MAX_HZ"),
        help="analysis band of the second panel (default: from config.toml)",
    )
    parser.add_argument(
        "--alpha", type=float, default=DEFAULT_ALPHA,
        help="family-wise false-positive rate across all bins and axes; "
             "above 0.01 it is a search mode: peaks passing only this "
             "threshold are drawn hollow",
    )
    parser.add_argument(
        "--baseline-window", type=float, default=None, metavar="HZ",
        help="width of the running-median baseline (default: from config.toml)",
    )
    return parser.parse_args(arguments)


def format_resolution(overview: Overview) -> str:
    low, high = overview.band_hz
    lines = [
        f"nperseg {overview.nperseg}, bin {overview.bin_width_hz:.3f} Hz: "
        f"{overview.segment_count} segments "
        f"({overview.effective_segments:.0f} independent)",
    ]
    listed = ", ".join(
        f"{peak.axis} {peak.frequency_hz:.1f} Hz" for peak in overview.full_band_peaks
    ) or "none"
    lines.append(f"  Full band, pooled: {listed}")
    reference = (
        f" ({overview.reference_threshold:.2f} at alpha {DEFAULT_ALPHA:g})"
        if overview.searching else ""
    )
    lines.append(
        f"  {low:g}-{high:g} Hz, pooled, z >= {overview.pooled_threshold:.2f} "
        f"at alpha {overview.alpha:g}{reference}:"
    )
    for peak in sorted(overview.pooled_peaks, key=lambda item: item.frequency_hz):
        mark = (
            "  search only"
            if not overview.passes_reference(peak) else ""
        )
        lines.append(
            f"    {peak.axis} {peak.frequency_hz:.2f} Hz  z {peak.z:.2f}  "
            f"+{peak.prominence_db:.2f} dB{mark}"
        )
    if not overview.pooled_peaks:
        lines.append("    none")
    narrow = sum(dip.narrow for dip in overview.pooled_dips)
    lines.append(
        f"  Dips below -{overview.pooled_threshold:.2f}: {narrow} narrow "
        f"(up to {NARROW_DIP_MAX_BINS} bins), "
        f"{len(overview.pooled_dips) - narrow} broad"
    )
    for dip in overview.pooled_dips:
        lines.append(
            f"    {dip.axis} {dip.frequency_hz:.2f} Hz  z {dip.z:.2f}  "
            f"{dip.width_bins} bins{'' if dip.narrow else ', broad'}"
        )
    if overview.effective_segments < MINIMUM_RELIABLE_PACKETS:
        lines.append(
            f"  WARNING: {overview.effective_segments:.0f} independent "
            f"segments is below {MINIMUM_RELIABLE_PACKETS}"
        )
    return "\n".join(lines) + "\n"


def main(arguments: list[str] | None = None) -> int:
    cli = parse_cli_arguments(arguments)
    try:
        captures = load_folder(cli.folder)
    except ValueError as error:
        print(f"error: {error}")
        return 1
    output = cli.output or STABLE_RESULTS_DIRECTORY / cli.folder.resolve().name
    for capture in captures:
        print(f"  {capture.name}: {capture.axes['x'].shape[0]} packets")
    for nperseg in dict.fromkeys(cli.nperseg):
        try:
            overview = build_overview(
                captures,
                tuple(cli.band) if cli.band is not None else None,
                cli.alpha,
                cli.baseline_window,
                int(nperseg),
            )
        except ValueError as error:
            print(f"nperseg {nperseg}: skipped: {error}")
            continue
        directory = output / resolution_directory_name(int(nperseg))
        figure = directory / "figure_overview.png"
        save_overview(figure, overview)
        report = format_resolution(overview)
        (directory / "overview.txt").write_text(
            report, encoding="utf-8", newline="\n",
        )
        print(report, end="")
        print(f"  Saved: {figure}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
