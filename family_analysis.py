"""Automatic data-driven frequency-family analysis over replay outputs.

This is an offline research/statistics layer. It reads the structured
replay outputs (``replay_runs.csv`` and ``replay_regions.csv``) produced by
``main.py --replay`` and groups trusted regions from many temporal windows
(virtual runs) into recurring frequency families without any prior
knowledge of the building frequencies.

It does not change the detector and does not introduce a new trusted gate.

Usage::

    python family_analysis.py replay_results/<raw_stem> [more dirs ...]
        [--output family_results/<name>]
        [--link-tolerance-hz 0.40] [--max-family-span-hz 0.80]

Method summary
--------------
* Observation = one trusted region in one virtual run.
* Families are built separately for each (axis, band).
* Pairwise distance between two observations combines both frequency
  descriptors that the detector produces:

      d = 0.5 * (|dFreq| + |dMed.Freq|)

  ``Freq`` is the session-recurrence centre, ``Med.Freq`` the Median PSD
  spectral maximum. Neither is a mandatory gate on its own, but two
  observations are declared incompatible (infinite distance) when
  * their session-peak ranges are separated by a gap larger than 1.5
    Welch bins (each range is padded by 0.75 bin), or
  * either |dFreq| or |dMed.Freq| exceeds ``max_family_span_hz``.
* Observations are grouped with agglomerative clustering cut at
  ``link_tolerance_hz``. The default average linkage tolerates the
  Welch-bin quantisation of ``Med.Freq`` (0.244 Hz at ODR 250) and the
  natural jitter of ``Freq`` within one physical structure, while the hard
  incompatibility rules above still cap every family: no two members may
  differ by more than ``max_family_span_hz`` in either descriptor, so a
  chain 3.2 -> 3.5 -> 3.8 -> 4.1 cannot merge into one family.
  ``--linkage complete`` bounds the family diameter by the link tolerance
  itself and is stricter (it tends to split one mode whose Med.Freq
  alternates between neighbouring bins).
* Within one family, one virtual run contributes at most one observation
  (the representative: highest support fraction, then highest Median
  prominence, then closest to the family centre). Other regions of the
  same run are kept in ``family_observations.csv`` with
  ``is_representative = 0``.

Independence caveat
-------------------
Nested virtual windows of one raw capture (4x4, 4x8, 4x16, ...) are built
from the same packets and are not independent samples. Therefore:
* temporal occupancy is computed separately per layout;
* capture recurrence is computed over independent raw captures;
* cross-scale consistency is reported as a separate profile, never as a
  pooled count of windows.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

FAMILY_RESULTS_DIRECTORY = Path("family_results")
WELCH_NPERSEG_DEFAULT = 1024
INCOMPATIBLE_DISTANCE = 1.0e6
# Each session-peak range is padded by this many Welch bins before the
# overlap test, so two ranges may be separated by up to 1.5 bins. Peaks on
# adjacent bins (one-bin gap) therefore always stay compatible, while a gap
# of two or more bins is rejected.
RANGE_PADDING_BINS = 0.75
LINKAGE_METHODS = ("average", "complete")

LAYOUT_ORDER = ("4x4", "4x8", "4x16", "4x32", "4x64",
                "8x4", "8x8", "8x16", "8x32")
CONTROLLED_PAIRS = (("4x8", "8x4"), ("4x16", "8x8"),
                    ("4x32", "8x16"), ("4x64", "8x32"))

AXIS_ORDER = {"X": 0, "Y": 1, "Z": 2}


# ==========================================================
# Data model
# ==========================================================


@dataclass(frozen=True)
class Window:
    capture: str
    mode: str
    virtual_run: int
    packets_per_session: int
    sessions_per_run: int
    packet_start: int
    packet_end: int
    packet_count: int
    duration_seconds: float

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.capture, self.mode, self.virtual_run)

    @property
    def packets_per_run(self) -> int:
        return self.packets_per_session * self.sessions_per_run


@dataclass
class Observation:
    observation_id: int
    capture: str
    source: str
    odr_hz: float
    frequency_tolerance_hz: float
    mode: str
    packets_per_session: int
    sessions_per_run: int
    virtual_run: int
    packet_start: int
    packet_end: int
    duration_seconds: float
    axis: str
    band: str
    freq_hz: float
    med_freq_hz: float
    support_n: int
    support_total: int
    support_fraction: float
    range_min_hz: float
    range_max_hz: float
    frequency_std_hz: float
    med_prom_db: float
    med_contrast_db: float
    band_contrast_db: float
    sources: int
    weight: float
    family_id: str = ""
    is_representative: bool = True
    regions_in_window: int = 1

    @property
    def window_key(self) -> tuple[str, str, int]:
        return (self.capture, self.mode, self.virtual_run)

    @property
    def packet_key(self) -> tuple[str, int, int]:
        return (self.capture, self.packet_start, self.packet_end)


@dataclass
class Family:
    family_id: str
    axis: str
    band: str
    observations: list[Observation]

    @property
    def representatives(self) -> list[Observation]:
        return [o for o in self.observations if o.is_representative]


# ==========================================================
# Loading
# ==========================================================


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def capture_name(source: str) -> str:
    return Path(source).stem


def load_windows(replay_directory: Path) -> list[Window]:
    rows = read_csv_rows(replay_directory / "replay_runs.csv")
    windows: dict[tuple[str, str, int], Window] = {}
    for row in rows:
        window = Window(
            capture=capture_name(row["source"]),
            mode=row["mode"],
            virtual_run=int(row["virtual_run"]),
            packets_per_session=int(row["packets_per_session"]),
            sessions_per_run=int(row["sessions_per_run"]),
            packet_start=int(row["packet_start"]),
            packet_end=int(row["packet_end"]),
            packet_count=int(row["packet_count"]),
            duration_seconds=float(row["duration_seconds"]),
        )
        windows.setdefault(window.key, window)
    return list(windows.values())


def load_observations(
    replay_directory: Path,
    first_id: int,
) -> list[Observation]:
    rows = read_csv_rows(replay_directory / "replay_regions.csv")
    observations = []
    for offset, row in enumerate(rows):
        observations.append(Observation(
            observation_id=first_id + offset,
            capture=capture_name(row["source"]),
            source=row["source"],
            odr_hz=float(row["odr_hz"]),
            frequency_tolerance_hz=float(row["frequency_tolerance_hz"]),
            mode=row["mode"],
            packets_per_session=int(row["packets_per_session"]),
            sessions_per_run=int(row["sessions_per_run"]),
            virtual_run=int(row["virtual_run"]),
            packet_start=int(row["packet_start"]),
            packet_end=int(row["packet_end"]),
            duration_seconds=float(row["duration_seconds"]),
            axis=row["axis"],
            band=row["band"],
            freq_hz=float(row["freq_hz"]),
            med_freq_hz=float(row["med_freq_hz"]),
            support_n=int(row["support_n"]),
            support_total=int(row["support_total"]),
            support_fraction=float(row["support_fraction"]),
            range_min_hz=float(row["range_min_hz"]),
            range_max_hz=float(row["range_max_hz"]),
            frequency_std_hz=float(row["frequency_std_hz"]),
            med_prom_db=float(row["med_prom_db"]),
            med_contrast_db=float(row["med_contrast_db"]),
            band_contrast_db=float(row["band_contrast_db"]),
            sources=int(row["sources"]),
            weight=float(row["weight"]),
        ))
    return observations


REPORT_HEADER_PATTERNS = {
    "odr_hz": re.compile(r"^ODR:\s*([\d.]+)\s*Hz"),
    "packets_per_session": re.compile(r"^Packets/session:\s*(\d+)"),
    "target_sessions": re.compile(r"^(?:Target sessions|Sessions):\s*(\d+)"),
    "frequency_tolerance_hz": re.compile(
        r"^Frequency tolerance:\s*([\d.]+)\s*Hz"
    ),
}
REPORT_PACKETS_PATTERN = re.compile(
    r"^Packets:\s*(\d+)\s+Duration:\s*([\d.]+)\s*s"
)
REPORT_TRUSTED_SECTION_PATTERN = re.compile(
    r"^Trusted frequency regions\s+[—-]\s+([XYZ])\s*$"
)
REPORT_TRUSTED_ROW_PATTERN = re.compile(
    r"^(?P<band>.+?)\s{2,}(?P<freq>[\d.]+)\s+(?P<support_n>\d+)/(?P<support_total>\d+)"
    r"\s+(?P<med>[\d.]+)\s+(?P<prom>-?[\d.]+)\s+(?P<contr>-?[\d.]+)"
    r"\s+(?P<band_contr>-?[\d.]+)\s+(?P<weight>[\d.]+)"
    r"\s+(?P<range_min>[\d.]+)\s*[–-]\s*(?P<range_max>[\d.]+)\s*$"
)
DEFAULT_TOLERANCE_BY_ODR = {250.0: 0.40, 125.0: 0.35, 62.5: 0.25}


def default_frequency_tolerance_hz(odr_hz: float) -> float:
    """Effective ODR tolerance for reports that predate the tolerance line.

    Reads config.toml next to this file when available, otherwise falls back
    to the documented ODR-dependent values.
    """
    key_by_odr = {250.0: "250", 125.0: "125", 62.5: "62p5"}
    key = key_by_odr.get(float(odr_hz))
    config_path = Path(__file__).with_name("config.toml")
    if key is not None and config_path.exists():
        try:
            import tomllib

            with config_path.open("rb") as config_file:
                config = tomllib.load(config_file)
            value = config["analysis"]["frequency_clustering"][
                f"frequency_tolerance_hz_{key}"
            ]
            return float(value)
        except (KeyError, OSError, ValueError):
            pass
    if float(odr_hz) in DEFAULT_TOLERANCE_BY_ODR:
        return DEFAULT_TOLERANCE_BY_ODR[float(odr_hz)]
    raise ValueError(
        f"No frequency tolerance known for ODR {odr_hz:g} Hz; "
        "pass --link-tolerance-hz"
    )


def parse_report(text: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Parse a measurement report (.txt) written by main.py.

    Returns the header fields and the rows of the "Trusted frequency
    regions" sections. Only trusted regions are used, matching the
    replay_regions.csv semantics.
    """
    header: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    current_axis: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        for name, pattern in REPORT_HEADER_PATTERNS.items():
            match = pattern.match(line)
            if match and name not in header:
                header[name] = float(match.group(1))
        match = REPORT_PACKETS_PATTERN.match(line)
        if match and "packet_count" not in header:
            header["packet_count"] = int(match.group(1))
            header["duration_seconds"] = float(match.group(2))
        section = REPORT_TRUSTED_SECTION_PATTERN.match(line)
        if section:
            current_axis = section.group(1)
            continue
        if current_axis is None:
            continue
        if not line.strip():
            current_axis = None
            continue
        if line.startswith("Band"):
            continue
        row_match = REPORT_TRUSTED_ROW_PATTERN.match(line)
        if row_match is None:
            continue
        rows.append({
            "axis": current_axis,
            "band": row_match.group("band").strip(),
            "freq_hz": float(row_match.group("freq")),
            "support_n": int(row_match.group("support_n")),
            "support_total": int(row_match.group("support_total")),
            "med_freq_hz": float(row_match.group("med")),
            "med_prom_db": float(row_match.group("prom")),
            "med_contrast_db": float(row_match.group("contr")),
            "band_contrast_db": float(row_match.group("band_contr")),
            "weight": float(row_match.group("weight")),
            "range_min_hz": float(row_match.group("range_min")),
            "range_max_hz": float(row_match.group("range_max")),
        })
    return header, rows


def load_report(
    report_path: Path,
    first_id: int,
    link_tolerance_hz: float | None,
) -> tuple[Window, list[Observation]]:
    """Build one window and its trusted observations from a .txt report.

    Reports without raw data cannot be replayed, but their trusted regions
    are still independent physical captures. They enter the analysis as
    one window of layout "<packets/session>x<sessions>" with
    ``frequency_std_hz`` unknown (NaN) and ``sources`` unknown (0).
    """
    header, rows = parse_report(report_path.read_text(encoding="utf-8"))
    required = ("odr_hz", "packets_per_session", "target_sessions",
                "packet_count", "duration_seconds")
    missing = [name for name in required if name not in header]
    if missing:
        raise ValueError(
            f"{report_path}: cannot parse report header fields {missing}"
        )
    odr_hz = float(header["odr_hz"])
    packets_per_session = int(header["packets_per_session"])
    sessions = int(header["target_sessions"])
    tolerance = header.get("frequency_tolerance_hz")
    if tolerance is None:
        tolerance = (
            link_tolerance_hz
            if link_tolerance_hz is not None
            else default_frequency_tolerance_hz(odr_hz)
        )
    capture = report_path.stem
    mode = f"{packets_per_session}x{sessions}"
    window = Window(
        capture=capture,
        mode=mode,
        virtual_run=1,
        packets_per_session=packets_per_session,
        sessions_per_run=sessions,
        packet_start=1,
        packet_end=int(header["packet_count"]),
        packet_count=int(header["packet_count"]),
        duration_seconds=float(header["duration_seconds"]),
    )
    observations = []
    for offset, row in enumerate(rows):
        observations.append(Observation(
            observation_id=first_id + offset,
            capture=capture,
            source=str(report_path.resolve()),
            odr_hz=odr_hz,
            frequency_tolerance_hz=float(tolerance),
            mode=mode,
            packets_per_session=packets_per_session,
            sessions_per_run=sessions,
            virtual_run=1,
            packet_start=window.packet_start,
            packet_end=window.packet_end,
            duration_seconds=window.duration_seconds,
            axis=row["axis"],
            band=row["band"],
            freq_hz=row["freq_hz"],
            med_freq_hz=row["med_freq_hz"],
            support_n=row["support_n"],
            support_total=row["support_total"],
            support_fraction=row["support_n"] / row["support_total"],
            range_min_hz=row["range_min_hz"],
            range_max_hz=row["range_max_hz"],
            frequency_std_hz=math.nan,
            med_prom_db=row["med_prom_db"],
            med_contrast_db=row["med_contrast_db"],
            band_contrast_db=row["band_contrast_db"],
            sources=0,
            weight=row["weight"],
        ))
    return window, observations


# ==========================================================
# Family construction
# ==========================================================


def pairwise_distance(
    first: Observation,
    second: Observation,
    max_family_span_hz: float,
    range_padding_hz: float,
) -> float:
    delta_freq = abs(first.freq_hz - second.freq_hz)
    delta_med = abs(first.med_freq_hz - second.med_freq_hz)
    if delta_freq > max_family_span_hz or delta_med > max_family_span_hz:
        return INCOMPATIBLE_DISTANCE
    overlap_low = max(
        first.range_min_hz - range_padding_hz,
        second.range_min_hz - range_padding_hz,
    )
    overlap_high = min(
        first.range_max_hz + range_padding_hz,
        second.range_max_hz + range_padding_hz,
    )
    if overlap_high < overlap_low:
        return INCOMPATIBLE_DISTANCE
    return 0.5 * (delta_freq + delta_med)


def cluster_observations(
    observations: list[Observation],
    link_tolerance_hz: float,
    max_family_span_hz: float,
    range_padding_hz: float,
    linkage_method: str = "average",
) -> list[list[Observation]]:
    if linkage_method not in LINKAGE_METHODS:
        raise ValueError(f"Unsupported linkage method: {linkage_method}")
    if len(observations) == 0:
        return []
    if len(observations) == 1:
        return [list(observations)]
    count = len(observations)
    condensed = []
    for i in range(count):
        for j in range(i + 1, count):
            condensed.append(pairwise_distance(
                observations[i],
                observations[j],
                max_family_span_hz,
                range_padding_hz,
            ))
    tree = linkage(np.asarray(condensed, dtype=float), method=linkage_method)
    labels = fcluster(tree, t=link_tolerance_hz, criterion="distance")
    groups: dict[int, list[Observation]] = defaultdict(list)
    for label, observation in zip(labels, observations):
        groups[int(label)].append(observation)
    return list(groups.values())


def representative_sort_key(
    observation: Observation,
    family_center_hz: float,
) -> tuple[float, float, float, int]:
    return (
        -observation.support_fraction,
        -observation.med_prom_db,
        abs(observation.freq_hz - family_center_hz),
        observation.observation_id,
    )


def assign_representatives(members: list[Observation]) -> None:
    center = float(np.median([o.freq_hz for o in members]))
    by_window: dict[tuple[str, str, int], list[Observation]] = defaultdict(list)
    for observation in members:
        by_window[observation.window_key].append(observation)
    for window_members in by_window.values():
        window_members.sort(key=lambda o: representative_sort_key(o, center))
        for index, observation in enumerate(window_members):
            observation.is_representative = index == 0
            observation.regions_in_window = len(window_members)


def build_families(
    observations: list[Observation],
    link_tolerance_hz: float,
    max_family_span_hz: float,
    range_padding_hz: float,
    linkage_method: str = "average",
) -> list[Family]:
    grouped: dict[tuple[str, str], list[Observation]] = defaultdict(list)
    for observation in observations:
        grouped[(observation.axis, observation.band)].append(observation)
    families = []
    for (axis, band), members in grouped.items():
        clusters = cluster_observations(
            members,
            link_tolerance_hz,
            max_family_span_hz,
            range_padding_hz,
            linkage_method,
        )
        for cluster in clusters:
            assign_representatives(cluster)
            families.append(Family(
                family_id="",
                axis=axis,
                band=band,
                observations=sorted(
                    cluster,
                    key=lambda o: (
                        o.capture, o.mode, o.virtual_run, o.freq_hz,
                    ),
                ),
            ))
    families.sort(key=lambda f: (
        AXIS_ORDER.get(f.axis, 99),
        float(np.median([o.freq_hz for o in f.representatives])),
    ))
    per_axis_counter: dict[str, int] = defaultdict(int)
    for family in families:
        per_axis_counter[family.axis] += 1
        family.family_id = (
            f"F-{family.axis}-{per_axis_counter[family.axis]:02d}"
        )
        for observation in family.observations:
            observation.family_id = family.family_id
    return families


# ==========================================================
# Statistics
# ==========================================================


def finite_values(values: list[float]) -> list[float]:
    return [float(v) for v in values if v is not None and math.isfinite(v)]


def median_or_none(values: list[float]) -> float | None:
    values = finite_values(values)
    if not values:
        return None
    return float(np.median(values))


def std_or_none(values: list[float]) -> float | None:
    values = finite_values(values)
    if not values:
        return None
    return float(np.std(values))


def mean_or_none(values: list[float]) -> float | None:
    values = finite_values(values)
    if not values:
        return None
    return float(np.mean(values))


def format_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    return value


def write_csv_rows(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    with path.open("x", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: format_value(row.get(key)) for key in fieldnames
            })


def windows_by_layout(windows: list[Window]) -> dict[str, list[Window]]:
    result: dict[str, list[Window]] = defaultdict(list)
    for window in windows:
        result[window.mode].append(window)
    return result


def layout_sort_key(mode: str) -> tuple[int, int, int]:
    packets_per_session, sessions = (int(v) for v in mode.split("x"))
    return (packets_per_session * sessions, packets_per_session, sessions)


def present_layouts(windows: list[Window]) -> list[str]:
    modes = sorted({w.mode for w in windows}, key=layout_sort_key)
    ordered = [m for m in LAYOUT_ORDER if m in modes]
    ordered.extend(m for m in modes if m not in LAYOUT_ORDER)
    return ordered


def build_scale_profile_rows(
    families: list[Family],
    windows: list[Window],
) -> list[dict[str, Any]]:
    by_layout = windows_by_layout(windows)
    rows = []
    for family in families:
        representatives = family.representatives
        for mode in present_layouts(windows):
            layout_windows = by_layout[mode]
            members = [o for o in representatives if o.mode == mode]
            present_keys = {o.window_key for o in members}
            sample = layout_windows[0]
            rows.append({
                "family_id": family.family_id,
                "axis": family.axis,
                "band": family.band,
                "mode": mode,
                "packets_per_session": sample.packets_per_session,
                "sessions_per_run": sample.sessions_per_run,
                "packets_per_run": sample.packets_per_run,
                "windows_total": len(layout_windows),
                "windows_present": len(present_keys),
                "occupancy": len(present_keys) / len(layout_windows),
                "captures_present": len({o.capture for o in members}),
                "freq_center_hz": median_or_none(
                    [o.freq_hz for o in members]
                ),
                "freq_std_hz": std_or_none([o.freq_hz for o in members]),
                "med_freq_center_hz": median_or_none(
                    [o.med_freq_hz for o in members]
                ),
                "med_freq_std_hz": std_or_none(
                    [o.med_freq_hz for o in members]
                ),
                "median_support_fraction": median_or_none(
                    [o.support_fraction for o in members]
                ),
                "median_med_prom_db": median_or_none(
                    [o.med_prom_db for o in members]
                ),
                "median_med_contrast_db": median_or_none(
                    [o.med_contrast_db for o in members]
                ),
                "median_band_contrast_db": median_or_none(
                    [o.band_contrast_db for o in members]
                ),
                "median_sigma_f_hz": median_or_none(
                    [o.frequency_std_hz for o in members]
                ),
            })
    return rows


def build_summary_rows(
    families: list[Family],
    windows: list[Window],
    scale_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    captures = sorted({w.capture for w in windows})
    layouts = present_layouts(windows)
    longest_layout = max(layouts, key=layout_sort_key) if layouts else ""
    shortest_layout = min(layouts, key=layout_sort_key) if layouts else ""
    scale_index: dict[tuple[str, str], dict[str, Any]] = {
        (row["family_id"], row["mode"]): row for row in scale_rows
    }
    rows = []
    for family in families:
        representatives = family.representatives
        freqs = [o.freq_hz for o in representatives]
        meds = [o.med_freq_hz for o in representatives]
        family_captures = sorted({o.capture for o in representatives})
        occupancies = {
            mode: scale_index[(family.family_id, mode)]["occupancy"]
            for mode in layouts
        }
        row = {
            "family_id": family.family_id,
            "axis": family.axis,
            "band": family.band,
            "n_windows": len(representatives),
            "n_regions": len(family.observations),
            "n_captures": len(family_captures),
            "captures_total": len(captures),
            "capture_recurrence": len(family_captures) / len(captures),
            "captures": ";".join(family_captures),
            "freq_center_hz": median_or_none(freqs),
            "freq_std_hz": std_or_none(freqs),
            "freq_min_hz": min(freqs),
            "freq_max_hz": max(freqs),
            "med_freq_center_hz": median_or_none(meds),
            "med_freq_std_hz": std_or_none(meds),
            "med_freq_min_hz": min(meds),
            "med_freq_max_hz": max(meds),
            "med_minus_freq_median_hz": median_or_none(
                [o.med_freq_hz - o.freq_hz for o in representatives]
            ),
            "range_envelope_min_hz": min(
                o.range_min_hz for o in representatives
            ),
            "range_envelope_max_hz": max(
                o.range_max_hz for o in representatives
            ),
            "median_support_fraction": median_or_none(
                [o.support_fraction for o in representatives]
            ),
            "median_med_prom_db": median_or_none(
                [o.med_prom_db for o in representatives]
            ),
            "median_med_contrast_db": median_or_none(
                [o.med_contrast_db for o in representatives]
            ),
            "median_band_contrast_db": median_or_none(
                [o.band_contrast_db for o in representatives]
            ),
            "median_sigma_f_hz": median_or_none(
                [o.frequency_std_hz for o in representatives]
            ),
            "occupancy_min": min(occupancies.values()) if occupancies else None,
            "occupancy_max": max(occupancies.values()) if occupancies else None,
            "shortest_layout": shortest_layout,
            "occupancy_shortest_layout": occupancies.get(shortest_layout),
            "longest_layout": longest_layout,
            "occupancy_longest_layout": occupancies.get(longest_layout),
            "layouts_present": ";".join(
                mode for mode in layouts if occupancies[mode] > 0
            ),
        }
        for mode in layouts:
            row[f"occ_{mode}"] = occupancies[mode]
        rows.append(row)
    return rows


def build_capture_matrix_rows(
    families: list[Family],
    windows: list[Window],
) -> list[dict[str, Any]]:
    captures = sorted({w.capture for w in windows})
    layouts = present_layouts(windows)
    windows_per_capture_layout: dict[tuple[str, str], int] = defaultdict(int)
    for window in windows:
        windows_per_capture_layout[(window.capture, window.mode)] += 1
    rows = []
    for family in families:
        for capture in captures:
            members = [
                o for o in family.representatives if o.capture == capture
            ]
            row: dict[str, Any] = {
                "family_id": family.family_id,
                "axis": family.axis,
                "band": family.band,
                "capture": capture,
                "present": len(members) > 0,
                "windows_present_all_layouts": len(members),
                "freq_center_hz": median_or_none(
                    [o.freq_hz for o in members]
                ),
                "med_freq_center_hz": median_or_none(
                    [o.med_freq_hz for o in members]
                ),
                "median_support_fraction": median_or_none(
                    [o.support_fraction for o in members]
                ),
                "median_med_prom_db": median_or_none(
                    [o.med_prom_db for o in members]
                ),
            }
            for mode in layouts:
                total = windows_per_capture_layout.get((capture, mode), 0)
                present = len({
                    o.window_key for o in members if o.mode == mode
                })
                row[f"occ_{mode}"] = present / total if total > 0 else None
            rows.append(row)
    return rows


def build_pair_comparison_rows(
    families: list[Family],
    windows: list[Window],
) -> list[dict[str, Any]]:
    by_layout = windows_by_layout(windows)
    rows = []
    for mode_a, mode_b in CONTROLLED_PAIRS:
        windows_a = {
            (w.capture, w.packet_start, w.packet_end): w
            for w in by_layout.get(mode_a, [])
        }
        windows_b = {
            (w.capture, w.packet_start, w.packet_end): w
            for w in by_layout.get(mode_b, [])
        }
        shared = sorted(set(windows_a) & set(windows_b))
        if not shared:
            continue
        for family in families:
            members_a = {
                o.packet_key: o
                for o in family.representatives if o.mode == mode_a
            }
            members_b = {
                o.packet_key: o
                for o in family.representatives if o.mode == mode_b
            }
            both = [k for k in shared if k in members_a and k in members_b]
            only_a = [
                k for k in shared if k in members_a and k not in members_b
            ]
            only_b = [
                k for k in shared if k in members_b and k not in members_a
            ]
            if not (both or only_a or only_b):
                continue

            def delta(attribute: str) -> float | None:
                return mean_or_none([
                    getattr(members_b[key], attribute)
                    - getattr(members_a[key], attribute)
                    for key in both
                ])

            rows.append({
                "family_id": family.family_id,
                "axis": family.axis,
                "band": family.band,
                "mode_a": mode_a,
                "mode_b": mode_b,
                "packets_per_window": windows_a[shared[0]].packet_count,
                "windows_compared": len(shared),
                "both": len(both),
                "only_a": len(only_a),
                "only_b": len(only_b),
                "neither": len(shared) - len(both) - len(only_a) - len(only_b),
                "occupancy_a": (len(both) + len(only_a)) / len(shared),
                "occupancy_b": (len(both) + len(only_b)) / len(shared),
                "mean_delta_freq_hz_b_minus_a": delta("freq_hz"),
                "mean_delta_med_freq_hz_b_minus_a": delta("med_freq_hz"),
                "mean_delta_support_fraction_b_minus_a": delta(
                    "support_fraction"
                ),
                "mean_delta_med_prom_db_b_minus_a": delta("med_prom_db"),
                "mean_delta_band_contrast_db_b_minus_a": delta(
                    "band_contrast_db"
                ),
                "mean_delta_range_width_hz_b_minus_a": mean_or_none([
                    (members_b[k].range_max_hz - members_b[k].range_min_hz)
                    - (members_a[k].range_max_hz - members_a[k].range_min_hz)
                    for k in both
                ]),
            })
    return rows


OBSERVATION_FIELDS = [
    "family_id", "observation_id", "capture", "source", "odr_hz",
    "frequency_tolerance_hz", "mode", "packets_per_session",
    "sessions_per_run", "virtual_run", "packet_start", "packet_end",
    "duration_seconds", "axis", "band", "freq_hz", "med_freq_hz",
    "range_min_hz", "range_max_hz", "frequency_std_hz", "support_n",
    "support_total", "support_fraction", "med_prom_db", "med_contrast_db",
    "band_contrast_db", "sources", "weight", "is_representative",
    "regions_in_window",
]


def build_observation_rows(families: list[Family]) -> list[dict[str, Any]]:
    rows = []
    for family in families:
        for observation in family.observations:
            row = {
                field: getattr(observation, field)
                for field in OBSERVATION_FIELDS
            }
            row["family_id"] = family.family_id
            rows.append(row)
    return rows


# ==========================================================
# Figures
# ==========================================================


def family_color_map(families: list[Family]) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    colors = plt.get_cmap("tab20").colors
    return {
        family.family_id: colors[index % len(colors)]
        for index, family in enumerate(families)
    }


def save_time_figure(
    path: Path,
    capture: str,
    families: list[Family],
    windows: list[Window],
    colors: dict[str, Any],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    capture_windows = [w for w in windows if w.capture == capture]
    layouts = present_layouts(capture_windows)
    axes_names = ("X", "Y", "Z")
    figure, panels = plt.subplots(
        len(layouts),
        len(axes_names),
        figsize=(16, 2.2 * len(layouts) + 1),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    packet_total = max(w.packet_end for w in capture_windows)
    for row_index, mode in enumerate(layouts):
        for column_index, axis_name in enumerate(axes_names):
            panel = panels[row_index][column_index]
            for window in capture_windows:
                if window.mode == mode:
                    panel.axvspan(
                        window.packet_start - 1,
                        window.packet_end,
                        color="0.93" if window.virtual_run % 2 else "0.98",
                        lw=0,
                    )
            for family in families:
                if family.axis != axis_name:
                    continue
                color = colors[family.family_id]
                for observation in family.observations:
                    if (
                        observation.capture != capture
                        or observation.mode != mode
                    ):
                        continue
                    alpha = 0.9 if observation.is_representative else 0.35
                    span = [observation.packet_start - 1, observation.packet_end]
                    panel.plot(
                        span,
                        [observation.med_freq_hz] * 2,
                        color=color, lw=2.5, alpha=alpha,
                        solid_capstyle="butt",
                    )
                    panel.plot(
                        span,
                        [observation.freq_hz] * 2,
                        color=color, lw=1.0, ls=":", alpha=alpha,
                    )
                    panel.fill_between(
                        span,
                        observation.range_min_hz,
                        observation.range_max_hz,
                        color=color, alpha=0.12, lw=0,
                    )
            panel.set_xlim(0, packet_total)
            panel.grid(True, alpha=0.3)
            if column_index == 0:
                panel.set_ylabel(f"{mode}\nHz")
            if row_index == 0:
                panel.set_title(f"Axis {axis_name}")
            if row_index == len(layouts) - 1:
                panel.set_xlabel("packet index")
    handles = []
    labels = []
    for family in families:
        if any(o.capture == capture for o in family.observations):
            center = float(np.median(
                [o.med_freq_hz for o in family.representatives]
            ))
            handles.append(
                Line2D([0], [0], color=colors[family.family_id], lw=3)
            )
            labels.append(f"{family.family_id} ~{center:.2f} Hz")
    figure.legend(
        handles, labels,
        loc="upper center",
        ncol=min(6, max(1, len(labels))),
        fontsize=8,
        bbox_to_anchor=(0.5, 0.0),
    )
    figure.suptitle(
        f"Frequency families over time: {capture}\n"
        "solid = Med.Freq, dotted = Freq, shaded = session-peak range",
    )
    figure.tight_layout(rect=(0, 0.0, 1, 0.97))
    figure.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(figure)


def save_occupancy_figure(
    path: Path,
    families: list[Family],
    windows: list[Window],
    scale_rows: list[dict[str, Any]],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layouts = present_layouts(windows)
    if not families or not layouts:
        return
    matrix = np.zeros((len(families), len(layouts)))
    index = {(r["family_id"], r["mode"]): r["occupancy"] for r in scale_rows}
    for i, family in enumerate(families):
        for j, mode in enumerate(layouts):
            matrix[i, j] = index.get((family.family_id, mode), 0.0)
    figure, panel = plt.subplots(
        figsize=(1.1 * len(layouts) + 4, 0.35 * len(families) + 2),
    )
    image = panel.imshow(
        matrix, cmap="viridis", vmin=0, vmax=1, aspect="auto",
    )
    panel.set_xticks(range(len(layouts)))
    panel.set_xticklabels(layouts)
    panel.set_yticks(range(len(families)))
    panel.set_yticklabels([
        f"{f.family_id}  "
        f"{np.median([o.med_freq_hz for o in f.representatives]):.2f} Hz"
        for f in families
    ], fontsize=8)
    for i in range(len(families)):
        for j in range(len(layouts)):
            value = matrix[i, j]
            panel.text(
                j, i, f"{value:.2f}",
                ha="center", va="center", fontsize=7,
                color="white" if value < 0.6 else "black",
            )
    figure.colorbar(image, ax=panel, label="temporal occupancy (per layout)")
    panel.set_title(
        "Family occupancy per layout (windows present / windows total)"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


def save_freq_vs_med_figure(
    path: Path,
    families: list[Family],
    colors: dict[str, Any],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    axes_names = ("X", "Y", "Z")
    figure, panels = plt.subplots(1, 3, figsize=(22, 6))
    for panel, axis_name in zip(panels, axes_names):
        low = math.inf
        high = -math.inf
        for family in families:
            if family.axis != axis_name:
                continue
            representatives = family.representatives
            freqs = [o.freq_hz for o in representatives]
            meds = [o.med_freq_hz for o in representatives]
            sizes = [20 + 80 * o.support_fraction for o in representatives]
            panel.scatter(
                freqs, meds, s=sizes, color=colors[family.family_id],
                alpha=0.7, edgecolor="k", linewidths=0.3,
                label=f"{family.family_id} (n={len(representatives)})",
            )
            low = min(low, min(freqs), min(meds))
            high = max(high, max(freqs), max(meds))
        if math.isfinite(low):
            panel.plot(
                [low - 0.2, high + 0.2], [low - 0.2, high + 0.2],
                color="0.5", lw=0.8, ls="--",
            )
            panel.legend(
                fontsize=6, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                borderaxespad=0.0, ncol=1 if len(panel.collections) <= 24 else 2,
            )
        panel.set_title(f"Axis {axis_name}: Freq vs Med.Freq")
        panel.set_xlabel("Freq (session recurrence centre), Hz")
        panel.set_ylabel("Med.Freq (Median PSD maximum), Hz")
        panel.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


# ==========================================================
# Driver
# ==========================================================


def resolve_output_directory(
    output: Path | None,
    replay_directories: list[Path],
) -> Path:
    if output is not None:
        return output
    if len(replay_directories) == 1:
        return FAMILY_RESULTS_DIRECTORY / replay_directories[0].name
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return FAMILY_RESULTS_DIRECTORY / f"multi_{stamp}"


def write_metadata(
    path: Path,
    replay_directories: list[Path],
    windows: list[Window],
    observations: list[Observation],
    families: list[Family],
    link_tolerance_hz: float,
    max_family_span_hz: float,
    range_padding_hz: float,
    tolerance_source: str,
    linkage_method: str,
) -> None:
    captures = sorted({w.capture for w in windows})
    layouts = present_layouts(windows)
    by_layout = windows_by_layout(windows)
    with path.open("x", encoding="utf-8", newline="\n") as metadata:
        metadata.write(
            "Frequency-family analysis (research layer, no detector changes)\n"
            f"Created: {datetime.now().isoformat(timespec='seconds')}\n"
            "\nInputs:\n"
        )
        for directory in replay_directories:
            metadata.write(f"  {directory.resolve()}\n")
        metadata.write(
            f"\nCaptures ({len(captures)}): {', '.join(captures)}\n"
            f"Windows (virtual runs): {len(windows)}\n"
            f"Trusted-region observations: {len(observations)}\n"
            f"Families: {len(families)}\n"
            "\nParameters:\n"
            f"  link_tolerance_hz = {link_tolerance_hz:.3f} "
            f"({tolerance_source})\n"
            f"  max_family_span_hz = {max_family_span_hz:.3f}\n"
            f"  range_padding_hz = {range_padding_hz:.4f} "
            f"({RANGE_PADDING_BINS:g} Welch bin per range)\n"
            f"  linkage = {linkage_method}\n"
            "\nMethod:\n"
            "  observation = one trusted region in one virtual run\n"
            "  families built per (axis, band)\n"
            "  distance d = 0.5 * (|dFreq| + |dMed.Freq|)\n"
            "  incompatible if session-peak ranges are separated by more\n"
            "    than 1.5 Welch bins\n"
            "  incompatible if |dFreq| or |dMed.Freq| > max_family_span_hz\n"
            f"  {linkage_method}-linkage clustering cut at link_tolerance_hz\n"
            "  one representative observation per family per virtual run\n"
            "    (max support fraction, then max Med.Prom, then closest Freq)\n"
            "\nFamily ids (F-<axis>-<nn>) are ordered by axis and Freq centre\n"
            "within this run only; they change when inputs change.\n"
            "\nIndependence notes:\n"
            "  nested layouts of one capture share packets and are NOT\n"
            "  independent samples; occupancy is reported per layout,\n"
            "  capture_recurrence counts independent raw captures,\n"
            "  windows_present_all_layouts is a descriptive pooled count.\n"
            "  frequency_std_hz of consolidated regions is the maximum\n"
            "  source-cluster sigma_f (inherited from replay); it is NaN\n"
            "  for observations taken from .txt reports (sources = 0).\n"
            "\nLayouts:\n"
        )
        for mode in layouts:
            per_capture = ", ".join(
                f"{capture}: "
                f"{sum(1 for w in by_layout[mode] if w.capture == capture)}"
                for capture in captures
            )
            metadata.write(
                f"  {mode}: {len(by_layout[mode])} windows ({per_capture})\n"
            )


def print_console_summary(
    summary_rows: list[dict[str, Any]],
    layouts: list[str],
) -> None:
    header = (
        f"{'family':<9}{'ax':<3}{'Freq':>7}{'Med.F':>7}{'win':>5}"
        f"{'cap':>6}{'supp':>6}{'prom':>6}{'bandC':>6}  "
        + " ".join(f"{mode:>5}" for mode in layouts)
    )
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        captures = f"{row['n_captures']}/{row['captures_total']}"
        print(
            f"{row['family_id']:<9}{row['axis']:<3}"
            f"{row['freq_center_hz']:>7.2f}{row['med_freq_center_hz']:>7.2f}"
            f"{row['n_windows']:>5}{captures:>6}"
            f"{row['median_support_fraction']:>6.2f}"
            f"{row['median_med_prom_db']:>6.2f}"
            f"{row['median_band_contrast_db']:>6.2f}  "
            + " ".join(f"{row[f'occ_{mode}']:>5.2f}" for mode in layouts)
        )


def run_family_analysis(
    replay_directories: list[Path],
    output_directory: Path,
    link_tolerance_hz: float | None,
    max_family_span_hz: float | None,
    welch_nperseg: int,
    linkage_method: str = "average",
) -> Path:
    windows: list[Window] = []
    observations: list[Observation] = []
    for source in replay_directories:
        if source.is_dir():
            if not (source / "replay_regions.csv").exists():
                raise FileNotFoundError(
                    f"{source} does not contain replay_regions.csv"
                )
            windows.extend(load_windows(source))
            observations.extend(
                load_observations(source, len(observations))
            )
        elif source.suffix.lower() == ".txt":
            window, report_observations = load_report(
                source, len(observations), link_tolerance_hz,
            )
            windows.append(window)
            observations.extend(report_observations)
        else:
            raise FileNotFoundError(
                f"{source} is neither a replay directory nor a .txt report"
            )
    if not windows:
        raise ValueError("No virtual runs found in the inputs")

    tolerances = sorted({o.frequency_tolerance_hz for o in observations})
    odrs = sorted({o.odr_hz for o in observations})
    if link_tolerance_hz is None:
        if len(tolerances) > 1:
            raise ValueError(
                "Inputs use different effective frequency tolerances "
                f"{tolerances}; pass --link-tolerance-hz explicitly"
            )
        link_tolerance_hz = tolerances[0] if tolerances else 0.40
        tolerance_source = "effective ODR tolerance from replay"
    else:
        tolerance_source = "command line"
    if max_family_span_hz is None:
        max_family_span_hz = 2.0 * link_tolerance_hz
    odr_hz = odrs[0] if odrs else 250.0
    range_padding_hz = RANGE_PADDING_BINS * odr_hz / welch_nperseg

    families = build_families(
        observations,
        link_tolerance_hz,
        max_family_span_hz,
        range_padding_hz,
        linkage_method,
    )

    output_directory.mkdir(parents=True, exist_ok=False)
    scale_rows = build_scale_profile_rows(families, windows)
    summary_rows = build_summary_rows(families, windows, scale_rows)
    capture_rows = build_capture_matrix_rows(families, windows)
    pair_rows = build_pair_comparison_rows(families, windows)
    observation_rows = build_observation_rows(families)
    layouts = present_layouts(windows)

    write_csv_rows(
        output_directory / "family_observations.csv",
        observation_rows,
        OBSERVATION_FIELDS,
    )
    summary_fields = [
        "family_id", "axis", "band", "n_windows", "n_regions", "n_captures",
        "captures_total", "capture_recurrence", "captures",
        "freq_center_hz", "freq_std_hz", "freq_min_hz", "freq_max_hz",
        "med_freq_center_hz", "med_freq_std_hz", "med_freq_min_hz",
        "med_freq_max_hz", "med_minus_freq_median_hz",
        "range_envelope_min_hz", "range_envelope_max_hz",
        "median_support_fraction", "median_med_prom_db",
        "median_med_contrast_db", "median_band_contrast_db",
        "median_sigma_f_hz", "occupancy_min", "occupancy_max",
        "shortest_layout", "occupancy_shortest_layout",
        "longest_layout", "occupancy_longest_layout", "layouts_present",
    ] + [f"occ_{mode}" for mode in layouts]
    write_csv_rows(
        output_directory / "family_summary.csv", summary_rows, summary_fields,
    )
    write_csv_rows(
        output_directory / "family_scale_profile.csv",
        scale_rows,
        [
            "family_id", "axis", "band", "mode", "packets_per_session",
            "sessions_per_run", "packets_per_run", "windows_total",
            "windows_present", "occupancy", "captures_present",
            "freq_center_hz", "freq_std_hz", "med_freq_center_hz",
            "med_freq_std_hz", "median_support_fraction",
            "median_med_prom_db", "median_med_contrast_db",
            "median_band_contrast_db", "median_sigma_f_hz",
        ],
    )
    write_csv_rows(
        output_directory / "family_capture_matrix.csv",
        capture_rows,
        [
            "family_id", "axis", "band", "capture", "present",
            "windows_present_all_layouts", "freq_center_hz",
            "med_freq_center_hz", "median_support_fraction",
            "median_med_prom_db",
        ] + [f"occ_{mode}" for mode in layouts],
    )
    write_csv_rows(
        output_directory / "family_pair_comparison.csv",
        pair_rows,
        [
            "family_id", "axis", "band", "mode_a", "mode_b",
            "packets_per_window", "windows_compared", "both", "only_a",
            "only_b", "neither", "occupancy_a", "occupancy_b",
            "mean_delta_freq_hz_b_minus_a",
            "mean_delta_med_freq_hz_b_minus_a",
            "mean_delta_support_fraction_b_minus_a",
            "mean_delta_med_prom_db_b_minus_a",
            "mean_delta_band_contrast_db_b_minus_a",
            "mean_delta_range_width_hz_b_minus_a",
        ],
    )
    write_metadata(
        output_directory / "family_metadata.txt",
        replay_directories,
        windows,
        observations,
        families,
        link_tolerance_hz,
        max_family_span_hz,
        range_padding_hz,
        tolerance_source,
        linkage_method,
    )

    colors = family_color_map(families)
    for capture in sorted({w.capture for w in windows}):
        save_time_figure(
            output_directory / f"figure_family_time_{capture}.png",
            capture,
            families,
            windows,
            colors,
        )
    save_occupancy_figure(
        output_directory / "figure_family_occupancy.png",
        families,
        windows,
        scale_rows,
    )
    save_freq_vs_med_figure(
        output_directory / "figure_family_freq_vs_med.png",
        families,
        colors,
    )

    print(
        f"Captures: {len({w.capture for w in windows})}, "
        f"windows: {len(windows)}, observations: {len(observations)}, "
        f"families: {len(families)}"
    )
    print(
        f"link_tolerance_hz={link_tolerance_hz:.2f} "
        f"max_family_span_hz={max_family_span_hz:.2f} "
        f"linkage={linkage_method}"
    )
    print_console_summary(summary_rows, layouts)
    print(f"Saved family analysis: {output_directory}")
    return output_directory


def parse_cli_arguments(
    arguments: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automatic frequency-family analysis over replay outputs",
    )
    parser.add_argument(
        "replay_directories",
        nargs="+",
        type=Path,
        help="replay result directories (containing replay_runs.csv and "
             "replay_regions.csv) and/or measurement .txt reports; a "
             "report without raw data enters as one window of its own "
             "layout",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--link-tolerance-hz",
        type=float,
        default=None,
        help="complete-linkage cut (default: effective ODR tolerance "
             "recorded in replay_regions.csv)",
    )
    parser.add_argument(
        "--max-family-span-hz",
        type=float,
        default=None,
        help="hard cap on |dFreq| and |dMed.Freq| between any two "
             "members (default: 2 x link tolerance)",
    )
    parser.add_argument(
        "--welch-nperseg",
        type=int,
        default=WELCH_NPERSEG_DEFAULT,
        help="used to derive the range padding (0.75 bin per range)",
    )
    parser.add_argument(
        "--linkage",
        choices=LINKAGE_METHODS,
        default="average",
        help="agglomerative linkage; average (default) is capped by "
             "max-family-span and the range gate, complete additionally "
             "bounds the family diameter by the link tolerance",
    )
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    cli = parse_cli_arguments(arguments)
    output_directory = resolve_output_directory(
        cli.output, cli.replay_directories,
    )
    run_family_analysis(
        cli.replay_directories,
        output_directory,
        cli.link_tolerance_hz,
        cli.max_family_span_hz,
        cli.welch_nperseg,
        cli.linkage,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
