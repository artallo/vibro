"""Run stable_spectrum.py on the control points and compare two runs.

Motivation
----------
A change to the method (a threshold, the baseline, the peak rules) is
judged by what it does on real records, before and after. Running every
folder at every resolution makes dozens of figures and leaves the list of
points in somebody's head. The control points are fixed in
``control_points.toml``, one entry per measuring point, and are described
in the vault note ``Данные/Контрольные точки``.

What it does
------------
1. Every point is analysed at one segment length, the one in
   ``[stable_spectrum] nperseg`` of ``config.toml`` (2048) unless
   ``--nperseg`` says otherwise: each run on its own, and all runs of the
   point together with ``--pool``. The method keys (``--alpha``,
   ``--baseline-window``, ``--min-distance``, ``--separation-sigma``,
   ``--band``) are passed to every analysis unchanged.
2. Probes of a point come from ``control_points.toml``: frequencies chosen
   before its records, judged with the honest probe threshold.
3. Everything goes to ``stable_results/_control/<variant>/``: the usual
   stable_spectrum.py output per point (``<point>/runs/``, ``<point>/pool/``)
   and one table of all peaks and probes, ``control_peaks.csv`` and
   ``control_probes.csv``, with ``control_summary.txt`` to read.
4. ``--compare A B`` matches the peaks of two variants on the same point,
   record and axis within ``--tolerance`` Hz and lists what was lost, what
   is new, and how z moved on the rest.

No building frequency is built in: probes live in ``control_points.toml``.

Usage::

    python control_run.py before
    python control_run.py after --separation-sigma 3
    python control_run.py --compare before after
    python control_run.py quick --points resp40_st2_f5 ervye_f1
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import re
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from stable_spectrum import (
    CONFIG_PATH,
    DEFAULT_ALPHA,
    DEFAULT_PROBE_TOLERANCE_HZ,
    STABLE_RESULTS_DIRECTORY,
    CaptureResult,
    ProbePlan,
    expand_raw_paths,
    load_default_nperseg,
    load_settings,
    parse_probe,
    run_single_resolution,
)

POINTS_PATH = Path(__file__).with_name("control_points.toml")
MAIN_PATH = Path(__file__).with_name("main.py")
DEFAULT_OLD_LAYOUTS = ["8x32", "8x8"]
CONTROL_DIRECTORY = STABLE_RESULTS_DIRECTORY / "_control"
POOL_RECORD = "pool"

PEAK_FIELDS = [
    "point", "record", "nperseg", "z_threshold", "axis", "frequency_hz",
    "z", "prominence_db", "persistent", "runs",
]
OLD_PEAK_FIELDS = [
    "point", "record", "layout", "nperseg", "axis", "frequency_hz",
    "prom_db", "support", "runs",
]
PROBE_FIELDS = [
    "point", "record", "nperseg", "axis", "requested_hz", "frequency_hz",
    "z", "threshold", "detected", "upper_bound_db",
]


@dataclass(frozen=True)
class ControlPoint:
    name: str
    folder: Path  # resolved against the folder of the points file
    probes: tuple[str, ...] = ()
    listed: str = ""  # the folder as written in the points file


@dataclass
class PointOutcome:
    point: ControlPoint
    runs: list[CaptureResult]
    pool: CaptureResult | None
    messages: list[str]


def load_points(path: Path) -> list[ControlPoint]:
    with path.open("rb") as points_file:
        entries = tomllib.load(points_file).get("point", [])
    points = []
    for entry in entries:
        name = str(entry["name"])
        if name in {point.name for point in points}:
            raise ValueError(f"{path.name}: point '{name}' is listed twice")
        listed = str(entry["folder"])
        points.append(ControlPoint(
            name, path.parent / listed,
            tuple(str(probe) for probe in entry.get("probes", [])), listed,
        ))
    if not points:
        raise ValueError(f"{path.name} lists no [[point]]")
    return points


def select_points(points: list[ControlPoint], names: list[str] | None) -> list[ControlPoint]:
    if not names:
        return points
    known = {point.name for point in points}
    unknown = [name for name in names if name not in known]
    if unknown:
        raise ValueError(
            f"unknown point(s) {', '.join(unknown)}; known: {', '.join(sorted(known))}"
        )
    return [point for point in points if point.name in names]


def single_nperseg(value: int | None) -> int:
    if value is not None:
        return value
    values = load_default_nperseg()
    if len(values) != 1:
        raise ValueError(
            f"config.toml [stable_spectrum] nperseg lists {values}; the control "
            "run uses one segment length: set one there or pass --nperseg"
        )
    return values[0]


def run_quietly(*arguments: Any) -> tuple[list[CaptureResult], list[str]]:
    """run_single_resolution without its per-record console lines.

    Skipped analyses and warnings are kept for the summary.
    """
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        results = run_single_resolution(*arguments)
    messages = [
        line.strip() for line in buffer.getvalue().splitlines()
        if "skipped" in line or line.startswith("WARNING")
    ]
    return results, messages


def analyse_point(point: ControlPoint, directory: Path, settings: Any, plan: ProbePlan) -> PointOutcome:
    try:
        raw_paths = expand_raw_paths([point.folder])
    except (OSError, ValueError) as error:
        raise ValueError(f"{point.name}: {error}") from error
    runs, messages = run_quietly(raw_paths, directory / "runs", settings, None, plan, False)
    pool = None
    if len(raw_paths) > 1:
        pooled, pool_messages = run_quietly(
            raw_paths, directory / POOL_RECORD, settings, None, plan, True,
        )
        pool = pooled[0] if pooled else None
        messages.extend(pool_messages)
    return PointOutcome(point, runs, pool, messages)


def records(outcome: PointOutcome) -> list[tuple[str, CaptureResult]]:
    labelled = [(result.capture, result) for result in outcome.runs]
    if outcome.pool is not None:
        labelled.append((POOL_RECORD, outcome.pool))
    return labelled


def peak_table(outcomes: list[PointOutcome]) -> list[dict[str, Any]]:
    rows = []
    for outcome in outcomes:
        for record, result in records(outcome):
            for peak in result.peaks:
                rows.append({
                    "point": outcome.point.name,
                    "record": record,
                    "nperseg": result.nperseg,
                    "z_threshold": round(result.z_threshold, 2),
                    "axis": peak.axis,
                    "frequency_hz": round(peak.frequency_hz, 3),
                    "z": round(peak.z, 2),
                    "prominence_db": round(peak.prominence_db, 2),
                    "persistent": int(peak.persistent),
                    "runs": peak.runs_label,
                })
    return rows


def probe_table(outcomes: list[PointOutcome]) -> list[dict[str, Any]]:
    rows = []
    for outcome in outcomes:
        for record, result in records(outcome):
            for probe in result.probes:
                rows.append({
                    "point": outcome.point.name,
                    "record": record,
                    "nperseg": result.nperseg,
                    "axis": probe.axis,
                    "requested_hz": round(probe.requested_hz, 3),
                    "frequency_hz": round(probe.frequency_hz, 3),
                    "z": round(probe.z, 2),
                    "threshold": round(result.probe_threshold, 2),
                    "detected": int(probe.detected),
                    "upper_bound_db": round(probe.upper_bound_db, 2),
                })
    return rows


def write_table(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def peak_text(result: CaptureResult) -> str:
    if not result.peaks:
        return "nothing above the noise"
    parts = []
    for peak in sorted(result.peaks, key=lambda item: item.frequency_hz):
        text = f"{peak.axis} {peak.frequency_hz:.2f} z {peak.z:.1f}"
        if not peak.persistent:
            text += "*"
        if peak.runs_label:
            text += f" {peak.runs_label}"
        parts.append(text)
    return ", ".join(parts)


def format_summary(variant: str, settings: Any, outcomes: list[PointOutcome]) -> str:
    lines = [
        f"Control run '{variant}'",
        f"Created: {datetime.now().isoformat(timespec='seconds')}",
        f"nperseg {settings.nperseg}   band {settings.band_hz[0]:g}-{settings.band_hz[1]:g} Hz   "
        f"alpha {settings.alpha:g}   baseline {settings.baseline_window_hz:g} Hz   "
        f"min distance {settings.min_distance_hz:g} Hz   "
        f"separation {settings.separation_sigma:g} sigma",
        "Peaks: axis Hz z; * not persistent across halves; k/N runs that find it alone.",
        "",
    ]
    for outcome in outcomes:
        point = outcome.point
        lines.append(f"{point.name}   {point.listed or point.folder.as_posix()}")
        for record, result in records(outcome):
            label = f"pool of {len(result.members)}" if record == POOL_RECORD else record
            lines.append(f"  {label:<36} z>={result.z_threshold:.2f}  {peak_text(result)}")
            for probe in result.probes:
                verdict = (
                    f"found at {probe.frequency_hz:.2f}, z {probe.z:.1f}"
                    if probe.detected else
                    f"not found, z {probe.z:.1f}, upper bound {probe.upper_bound_db:.2f} dB"
                )
                lines.append(
                    f"  {'':<36}   probe {probe.axis} {probe.requested_hz:g}: {verdict} "
                    f"(z>={result.probe_threshold:.2f})"
                )
        for message in outcome.messages:
            lines.append(f"  ! {message}")
        lines.append("")
    return "\n".join(lines)


def run_variant(cli: argparse.Namespace) -> int:
    variant = cli.variant
    if Path(variant).name != variant or variant in {".", ".."}:
        print(f"error: variant '{variant}' must be a plain folder name")
        return 1
    try:
        points = select_points(load_points(cli.points_file), cli.points)
        nperseg = single_nperseg(cli.nperseg)
        settings = load_settings(
            cli.alpha, tuple(cli.band) if cli.band else None,
            cli.baseline_window, cli.min_distance, cli.separation_sigma,
        )
        settings = replace(settings, nperseg=nperseg, noverlap=nperseg // 2)
        plans = {
            point.name: ProbePlan(requests=tuple(parse_probe(text) for text in point.probes))
            for point in points
        }
    except (OSError, ValueError, KeyError, argparse.ArgumentTypeError) as error:
        print(f"error: {error}")
        return 1

    output = cli.output / variant
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    outcomes = []
    for point in points:
        try:
            outcome = analyse_point(point, output / point.name, settings, plans[point.name])
        except ValueError as error:
            print(f"error: {error}")
            return 1
        outcomes.append(outcome)
        found = sum(len(result.peaks) for _, result in records(outcome))
        print(f"{point.name}: {len(outcome.runs)} run(s), {found} peak(s) in all records")

    write_table(output / "control_peaks.csv", PEAK_FIELDS, peak_table(outcomes))
    write_table(output / "control_probes.csv", PROBE_FIELDS, probe_table(outcomes))
    summary = format_summary(variant, settings, outcomes)
    (output / "control_summary.txt").write_text(summary, encoding="utf-8", newline="\n")
    print(f"Saved control run: {output}")
    return 0


# ==========================================================
# The old detector of main.py
# ==========================================================

def old_detector_config(text: str, nperseg: int | None) -> tuple[str, int, int]:
    """config.toml with [welch] nperseg and noverlap set; nothing else changes.

    Returns the text and the segment length and overlap it holds. Without
    ``nperseg`` the configured values stay.
    """
    lines = text.splitlines(keepends=True)
    start = next(
        (i for i, line in enumerate(lines) if line.strip() == "[welch]"), None,
    )
    if start is None:
        raise ValueError("config.toml has no [welch] section")
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].lstrip().startswith("[")),
        len(lines),
    )
    values = {}
    for i in range(start + 1, end):
        match = re.match(r"\s*(nperseg|noverlap)\s*=\s*(\d+)", lines[i])
        if match:
            values[match.group(1)] = (i, int(match.group(2)))
    if set(values) != {"nperseg", "noverlap"}:
        raise ValueError("[welch] of config.toml needs nperseg and noverlap")
    if nperseg is None:
        return text, values["nperseg"][1], values["noverlap"][1]
    for key, value in (("nperseg", nperseg), ("noverlap", nperseg // 2)):
        line = lines[values[key][0]]
        newline = line[len(line.rstrip("\r\n")):]
        lines[values[key][0]] = f"{key} = {value}{newline}"
    return "".join(lines), nperseg, nperseg // 2


def set_median_prominence(text: str, value: float | None) -> tuple[str, float]:
    """config.toml with the Med.Prom threshold of trusted regions set.

    Only ``min_median_prominence_db`` of ``[visualization.trusted_frequency]``
    changes. Returns the text and the threshold it holds.
    """
    lines = text.splitlines(keepends=True)
    start = next(
        (i for i, line in enumerate(lines)
         if line.strip() == "[visualization.trusted_frequency]"),
        None,
    )
    if start is None:
        raise ValueError("config.toml has no [visualization.trusted_frequency] section")
    for i in range(start + 1, len(lines)):
        if lines[i].lstrip().startswith("["):
            break
        match = re.match(r"\s*min_median_prominence_db\s*=\s*([\d.]+)", lines[i])
        if match:
            if value is None:
                return text, float(match.group(1))
            newline = lines[i][len(lines[i].rstrip("\r\n")):]
            lines[i] = f"min_median_prominence_db = {float(value)!r}{newline}"
            return "".join(lines), value
    raise ValueError("[visualization.trusted_frequency] needs min_median_prominence_db")


LAYOUT_TABLE = "[visualization.trusted_frequency.min_median_prominence_db_by_layout]"


def parse_threshold(text: str) -> tuple[str | None, float]:
    """``2.0`` is the default threshold, ``8x16=2.95`` the one of a layout."""
    layout, _, value = text.rpartition("=")
    if layout and not re.fullmatch(r"[1-9][0-9]*x[1-9][0-9]*", layout):
        raise argparse.ArgumentTypeError(f"'{text}': the layout must look like 8x16")
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"'{text}' is not 2.0 or 8x16=2.95") from error
    if number < 0:
        raise argparse.ArgumentTypeError(f"'{text}' must not be negative")
    return (layout or None), number


def set_layout_thresholds(text: str, thresholds: dict[str, float]) -> str:
    """config.toml with these entries in the table of thresholds by layout.

    Entries already there for other layouts stay; the table is added at the
    end when the file has none.
    """
    if not thresholds:
        return text
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines(keepends=True)
    start = next((i for i, line in enumerate(lines) if line.strip() == LAYOUT_TABLE), None)
    if start is None:
        if lines and not lines[-1].endswith(("\n", "\r")):
            lines[-1] += newline
        lines += [newline, LAYOUT_TABLE + newline]
        start = len(lines) - 1
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].lstrip().startswith("[")),
        len(lines),
    )
    for layout, value in thresholds.items():
        entry = f'"{layout}" = {float(value)!r}{newline}'
        found = next(
            (i for i in range(start + 1, end)
             if re.match(rf'\s*"?{re.escape(layout)}"?\s*=', lines[i])),
            None,
        )
        if found is None:
            lines.insert(start + 1, entry)
            end += 1
        else:
            lines[found] = entry
    return "".join(lines)


def applied_thresholds(text: str, layouts: list[str]) -> str:
    """What each layout of a run will use, as main.py resolves it."""
    trusted = tomllib.loads(text)["visualization"]["trusted_frequency"]
    table = trusted.get("min_median_prominence_db_by_layout", {})
    return ", ".join(
        f"{layout} {table[layout]:g}" if layout in table
        else f"{layout} {trusted['min_median_prominence_db']:g} (default)"
        for layout in layouts
    )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def group_by_frequency(
    items: list[dict[str, Any]], tolerance_hz: float,
) -> list[list[dict[str, Any]]]:
    """Regions of one axis within the tolerance of their group's mean."""
    groups: list[list[dict[str, Any]]] = []
    for item in sorted(items, key=lambda row: (row["axis"], row["frequency_hz"])):
        last = groups[-1] if groups else None
        if (
            last is not None and last[0]["axis"] == item["axis"]
            and abs(item["frequency_hz"] - np.mean([row["frequency_hz"] for row in last]))
            <= tolerance_hz
        ):
            last.append(item)
        else:
            groups.append([item])
    return groups


def summarise_group(group: list[dict[str, Any]], found: int, total: int) -> dict[str, Any]:
    return {
        "axis": group[0]["axis"],
        "frequency_hz": round(float(np.mean([row["frequency_hz"] for row in group])), 3),
        "prom_db": round(float(np.mean([row["prom_db"] for row in group])), 2),
        "support": "",
        "runs": f"{found}/{total}",
    }


def old_point_rows(
    point: ControlPoint, raw_paths: list[Path], replay_root: Path,
    layouts: list[str], nperseg: int, tolerance_hz: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Trusted regions of every run and layout, and the summary lines."""
    rows: list[dict[str, Any]] = []
    lines = [f"{point.name}   {point.listed or point.folder.as_posix()}"]
    whole: dict[str, list[dict[str, Any]]] = {layout: [] for layout in layouts}
    # Per layout: runs of that layout in all records, and those with a region.
    hit_counts = {layout: [0, 0] for layout in layouts}
    # Per layout: the highest Med.Prom of a trusted region. With
    # --median-prominence 0 every region with enough support is trusted, so
    # on noise this is the threshold a layout needs to stay clean.
    strongest: dict[str, float] = {}
    for raw_path in raw_paths:
        folder = replay_root / raw_path.stem
        regions = read_csv_rows(folder / "replay_regions.csv")
        runs = read_csv_rows(folder / "replay_runs.csv")
        lines.append(f"  {raw_path.stem}")
        for layout in layouts:
            parts = sorted({int(row["virtual_run"]) for row in runs if row["mode"] == layout})
            found = [
                {
                    "axis": row["axis"],
                    "frequency_hz": float(row["med_freq_hz"]),
                    "prom_db": float(row["med_prom_db"]),
                    "support": f"{row['support_n']}/{row['support_total']}",
                    "part": int(row["virtual_run"]),
                }
                for row in regions if row["mode"] == layout
            ]
            base = {"point": point.name, "layout": layout, "nperseg": nperseg}
            if not parts:
                lines.append(f"    {layout:<6} record too short for this layout")
                continue
            hit_counts[layout][0] += len(parts)
            hit_counts[layout][1] += len({item["part"] for item in found})
            if found:
                strongest[layout] = max(
                    strongest.get(layout, 0.0), max(item["prom_db"] for item in found),
                )
            if len(parts) == 1:
                kind = "full"
                entries = [
                    {
                        "axis": item["axis"],
                        "frequency_hz": round(item["frequency_hz"], 3),
                        "prom_db": round(item["prom_db"], 2),
                        "support": item["support"],
                        "runs": "",
                    }
                    for item in sorted(found, key=lambda row: (row["axis"], row["frequency_hz"]))
                ]
                text = ", ".join(
                    f"{item['axis']} {item['frequency_hz']:.2f} ({item['support']}, "
                    f"{item['prom_db']:.1f} dB)" for item in entries
                )
                whole[layout].extend({**item, "source": raw_path.stem} for item in found)
            else:
                kind = "parts"
                entries = [
                    summarise_group(group, len({row["part"] for row in group}), len(parts))
                    for group in group_by_frequency(found, tolerance_hz)
                ]
                text = ", ".join(
                    f"{item['axis']} {item['frequency_hz']:.2f} {item['runs']}" for item in entries
                )
            label = f"{layout} ({len(parts)} run{'s' if len(parts) > 1 else ''})"
            lines.append(f"    {label:<14} {text or 'no trusted region'}")
            rows.extend(
                {**base, "record": f"{raw_path.stem} {kind}", **entry} for entry in entries
            )
    lines.append(
        "  runs with a trusted region: "
        + ", ".join(
            f"{layout} {hits}/{total}" for layout, (total, hits) in hit_counts.items() if total
        )
    )
    if strongest:
        lines.append(
            "  strongest trusted Med.Prom: "
            + ", ".join(f"{layout} {value:.2f} dB" for layout, value in strongest.items())
        )
    if len(raw_paths) > 1:
        for layout, found in whole.items():
            if not found:
                continue
            entries = [
                summarise_group(group, len({row["source"] for row in group}), len(raw_paths))
                for group in group_by_frequency(found, tolerance_hz)
            ]
            lines.append(
                f"  across runs, {layout}: "
                + ", ".join(f"{item['axis']} {item['frequency_hz']:.2f} {item['runs']}" for item in entries)
            )
            rows.extend(
                {"point": point.name, "layout": layout, "nperseg": nperseg,
                 "record": "across runs full", **entry}
                for entry in entries
            )
    return rows, lines


def run_old_variant(cli: argparse.Namespace) -> int:
    variant = cli.variant
    if Path(variant).name != variant or variant in {".", ".."}:
        print(f"error: variant '{variant}' must be a plain folder name")
        return 1
    try:
        points = select_points(load_points(cli.points_file), cli.points)
        config_text, nperseg, noverlap = old_detector_config(
            CONFIG_PATH.read_text(encoding="utf-8"), cli.nperseg,
        )
        defaults = [value for layout, value in cli.median_prominence or [] if layout is None]
        if len(defaults) > 1:
            raise ValueError("--median-prominence takes one default value")
        config_text, _ = set_median_prominence(config_text, defaults[0] if defaults else None)
        config_text = set_layout_thresholds(config_text, {
            layout: value for layout, value in cli.median_prominence or [] if layout
        })
        thresholds_text = applied_thresholds(config_text, cli.layouts)
        point_paths = {}
        for point in points:
            try:
                point_paths[point.name] = expand_raw_paths([point.folder])
            except (OSError, ValueError) as error:
                raise ValueError(f"{point.name}: {error}") from error
    except (OSError, ValueError) as error:
        print(f"error: {error}")
        return 1

    output = cli.output / variant
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    config_path = output / "config.toml"
    config_path.write_text(config_text, encoding="utf-8")
    all_rows = []
    lines = [
        f"Old detector (main.py --replay) control run '{variant}'",
        f"Created: {datetime.now().isoformat(timespec='seconds')}",
        f"Welch nperseg {nperseg}, noverlap {noverlap}   layouts {' '.join(cli.layouts)}",
        f"Med.Prom thresholds, dB: {thresholds_text}",
        "Trusted regions at the spectral maximum of the median PSD (Med.Freq):",
        "one run of a layout: support n/N sessions and Med.Prom; several runs:",
        "in how many of them the region is trusted, grouped within "
        f"{cli.tolerance:g} Hz.",
        "",
    ]
    for point in points:
        replay_root = output / point.name
        for raw_path in point_paths[point.name]:
            completed = subprocess.run(
                [
                    sys.executable, str(MAIN_PATH), "--replay", str(raw_path),
                    "--virtual-mode", *cli.layouts,
                    "--config", str(config_path), "--replay-root", str(replay_root),
                ],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
            replay_root.mkdir(parents=True, exist_ok=True)
            (replay_root / f"{raw_path.stem}_replay.log").write_text(
                completed.stdout + completed.stderr, encoding="utf-8",
            )
            if completed.returncode != 0:
                print(f"error: main.py --replay failed on {raw_path}, see the log in {replay_root}")
                return 1
        rows, point_lines = old_point_rows(
            point, point_paths[point.name], replay_root, cli.layouts, nperseg, cli.tolerance,
        )
        all_rows.extend(rows)
        lines.extend(point_lines + [""])
        print(f"{point.name}: {len(point_paths[point.name])} run(s) replayed")

    write_table(output / "control_old_peaks.csv", OLD_PEAK_FIELDS, all_rows)
    (output / "control_summary.txt").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"Saved old detector control run: {output}")
    return 0


# ==========================================================
# Comparison of two variants
# ==========================================================

PEAK_TABLES = {
    "control_peaks.csv": ("z", "z"),
    "control_old_peaks.csv": ("prom_db", "Med.Prom dB"),
}


def read_peaks(directory: Path) -> tuple[list[dict[str, Any]], str, str]:
    """Peaks of one run, its table name and what the value column holds.

    The value goes to ``z`` either way, so both kinds compare alike.
    """
    for name, (column, label) in PEAK_TABLES.items():
        path = directory / name
        if path.exists():
            with path.open(encoding="utf-8", newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))
            for row in rows:
                row["frequency_hz"] = float(row["frequency_hz"])
                row["z"] = float(row[column])
            return rows, name, label
    raise OSError(f"no {' or '.join(PEAK_TABLES)} in {directory}")


def match_peaks(
    before: list[dict[str, Any]], after: list[dict[str, Any]], tolerance_hz: float,
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Pair peaks of one record and axis, nearest first, within the tolerance."""
    candidates = []
    for i, old in enumerate(before):
        for j, new in enumerate(after):
            same = all(old[key] == new[key] for key in ("point", "record", "axis"))
            distance = abs(old["frequency_hz"] - new["frequency_hz"])
            if same and distance <= tolerance_hz:
                candidates.append((distance, i, j))
    used_before: set[int] = set()
    used_after: set[int] = set()
    pairs = []
    for _, i, j in sorted(candidates):
        if i in used_before or j in used_after:
            continue
        used_before.add(i)
        used_after.add(j)
        pairs.append((before[i], after[j]))
    lost = [row for i, row in enumerate(before) if i not in used_before]
    new = [row for j, row in enumerate(after) if j not in used_after]
    return pairs, lost, new


def format_comparison(
    name_a: str, name_b: str,
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    lost: list[dict[str, Any]], new: list[dict[str, Any]],
    skipped: list[str], tolerance_hz: float, value_label: str = "z",
) -> str:
    def where(row: dict[str, Any]) -> str:
        return f"{row['point']:<16} {row['record']:<36}"

    def peak(row: dict[str, Any]) -> str:
        runs = f" {row['runs']}" if row.get("runs") else ""
        return f"{row['axis']} {row['frequency_hz']:.2f} {value_label} {row['z']:.1f}{runs}"

    order = lambda row: (row["point"], row["record"], row["axis"], row["frequency_hz"])
    lines = [
        f"Control runs compared: '{name_a}' -> '{name_b}'",
        f"Peaks matched on the same point, record and axis within {tolerance_hz:g} Hz.",
        f"Kept {len(pairs)}, lost {len(lost)}, new {len(new)}.",
        "",
    ]
    if skipped:
        lines.append("Points in only one of the runs, not compared: " + ", ".join(skipped))
        lines.append("")
    lines.append(f"Lost (in '{name_a}' only):")
    lines += [f"  {where(row)} {peak(row)}" for row in sorted(lost, key=order)] or ["  none"]
    lines.append("")
    lines.append(f"New (in '{name_b}' only):")
    lines += [f"  {where(row)} {peak(row)}" for row in sorted(new, key=order)] or ["  none"]
    lines.append("")
    lines.append(f"Kept, {value_label} before -> after:")
    for old, fresh in sorted(pairs, key=lambda pair: order(pair[0])):
        shift = fresh["frequency_hz"] - old["frequency_hz"]
        moved = f"  ({shift:+.2f} Hz)" if abs(shift) >= 0.005 else ""
        runs = ""
        if old.get("runs") or fresh.get("runs"):
            runs = f"  runs {old.get('runs') or '-'} -> {fresh.get('runs') or '-'}"
        lines.append(
            f"  {where(old)} {old['axis']} {old['frequency_hz']:.2f}  "
            f"{value_label} {old['z']:.1f} -> {fresh['z']:.1f}{moved}{runs}"
        )
    return "\n".join(lines) + "\n"


def compare_variants(cli: argparse.Namespace) -> int:
    name_a, name_b = cli.compare
    try:
        before, table_a, label = read_peaks(cli.output / name_a)
        after, table_b, _ = read_peaks(cli.output / name_b)
    except OSError as error:
        print(f"error: {error}")
        return 1
    if table_a != table_b:
        print(f"error: '{name_a}' and '{name_b}' are runs of different detectors")
        return 1
    summary_points = []
    for name in (name_a, name_b):
        directory = cli.output / name
        summary_points.append({path.name for path in directory.iterdir() if path.is_dir()})
    common = summary_points[0] & summary_points[1]
    skipped = sorted(summary_points[0] ^ summary_points[1])
    pairs, lost, new = match_peaks(
        [row for row in before if row["point"] in common],
        [row for row in after if row["point"] in common],
        cli.tolerance,
    )
    text = format_comparison(name_a, name_b, pairs, lost, new, skipped, cli.tolerance, label)
    path = cli.output / f"compare_{name_a}_{name_b}.txt"
    path.write_text(text, encoding="utf-8", newline="\n")
    print(text)
    print(f"Saved: {path}")
    return 0


def parse_cli_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="stable_spectrum.py on the control points, and the "
                    "comparison of two such runs",
    )
    parser.add_argument(
        "variant", nargs="?", default=None,
        help="name of this run, e.g. before or after; its results go to "
             "<output>/<variant>/, replacing an earlier run of that name",
    )
    parser.add_argument(
        "--compare", nargs=2, default=None, metavar=("A", "B"),
        help="compare the peaks of two earlier runs instead of running",
    )
    parser.add_argument("--output", type=Path, default=CONTROL_DIRECTORY)
    parser.add_argument("--points-file", type=Path, default=POINTS_PATH)
    parser.add_argument(
        "--points", nargs="+", default=None, metavar="NAME",
        help="run only these points of the list",
    )
    parser.add_argument(
        "--nperseg", type=int, default=None, metavar="SAMPLES",
        help="one segment length (default: [stable_spectrum] nperseg in "
             "config.toml, which must hold one value; with --old-detector "
             "[welch] nperseg, and noverlap becomes half of it)",
    )
    parser.add_argument(
        "--old-detector", action="store_true",
        help="run the old detector of main.py (--replay) instead of "
             "stable_spectrum.py and table its trusted regions",
    )
    parser.add_argument(
        "--median-prominence", type=parse_threshold, nargs="+", default=None,
        metavar="DB|PxS=DB",
        help="--old-detector: Med.Prom a trusted region needs; a plain value "
             "sets the default, 8x16=2.95 the threshold of one layout "
             "(default: config.toml)",
    )
    parser.add_argument(
        "--layouts", nargs="+", default=None, metavar="PxS",
        help="--old-detector: replay layouts, packets per session x sessions "
             f"per run (default {' '.join(DEFAULT_OLD_LAYOUTS)})",
    )
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument(
        "--band", type=float, nargs=2, default=None, metavar=("MIN_HZ", "MAX_HZ"),
    )
    parser.add_argument("--baseline-window", type=float, default=None, metavar="HZ")
    parser.add_argument("--min-distance", type=float, default=None, metavar="HZ")
    parser.add_argument("--separation-sigma", type=float, default=None, metavar="K")
    parser.add_argument(
        "--tolerance", type=float, default=DEFAULT_PROBE_TOLERANCE_HZ, metavar="HZ",
        help="--compare: how far apart one peak may be in the two runs "
             f"(default {DEFAULT_PROBE_TOLERANCE_HZ:g} Hz)",
    )
    cli = parser.parse_args(arguments)
    if (cli.variant is None) == (cli.compare is None):
        parser.error("give either a variant name to run or --compare A B")
    if cli.old_detector:
        unused = [
            key for key, value in (
                ("--alpha", cli.alpha != DEFAULT_ALPHA), ("--band", cli.band),
                ("--baseline-window", cli.baseline_window),
                ("--min-distance", cli.min_distance),
                ("--separation-sigma", cli.separation_sigma),
            ) if value
        ]
        if unused:
            parser.error(f"{', '.join(unused)}: not used by the old detector")
        cli.layouts = cli.layouts or list(DEFAULT_OLD_LAYOUTS)
    elif cli.layouts or cli.median_prominence is not None:
        parser.error("--layouts and --median-prominence need --old-detector")
    return cli


def main(arguments: list[str] | None = None) -> int:
    cli = parse_cli_arguments(arguments)
    if cli.compare:
        return compare_variants(cli)
    if cli.old_detector:
        return run_old_variant(cli)
    return run_variant(cli)


if __name__ == "__main__":
    sys.exit(main())
