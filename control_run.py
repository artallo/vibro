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
import shutil
import sys
import tomllib
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from stable_spectrum import (
    DEFAULT_ALPHA,
    DEFAULT_PROBE_TOLERANCE_HZ,
    STABLE_RESULTS_DIRECTORY,
    CaptureResult,
    ProbePlan,
    load_default_nperseg,
    load_settings,
    parse_probe,
    run_single_resolution,
)

POINTS_PATH = Path(__file__).with_name("control_points.toml")
CONTROL_DIRECTORY = STABLE_RESULTS_DIRECTORY / "_control"
POOL_RECORD = "pool"

PEAK_FIELDS = [
    "point", "record", "nperseg", "z_threshold", "axis", "frequency_hz",
    "z", "prominence_db", "persistent", "runs",
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
    raw_paths = sorted(point.folder.glob("*_raw.npz"))
    if not raw_paths:
        raise ValueError(f"{point.name}: no *_raw.npz in {point.folder}")
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
# Comparison of two variants
# ==========================================================

def read_peaks(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    for row in rows:
        row["frequency_hz"] = float(row["frequency_hz"])
        row["z"] = float(row["z"])
    return rows


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
    skipped: list[str], tolerance_hz: float,
) -> str:
    def where(row: dict[str, Any]) -> str:
        return f"{row['point']:<16} {row['record']:<36}"

    def peak(row: dict[str, Any]) -> str:
        runs = f" {row['runs']}" if row.get("runs") else ""
        return f"{row['axis']} {row['frequency_hz']:.2f} z {row['z']:.1f}{runs}"

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
    lines.append("Kept, z before -> after:")
    for old, fresh in sorted(pairs, key=lambda pair: order(pair[0])):
        shift = fresh["frequency_hz"] - old["frequency_hz"]
        moved = f"  ({shift:+.2f} Hz)" if abs(shift) >= 0.005 else ""
        runs = ""
        if old.get("runs") or fresh.get("runs"):
            runs = f"  runs {old.get('runs') or '-'} -> {fresh.get('runs') or '-'}"
        lines.append(
            f"  {where(old)} {old['axis']} {old['frequency_hz']:.2f}  "
            f"z {old['z']:.1f} -> {fresh['z']:.1f}{moved}{runs}"
        )
    return "\n".join(lines) + "\n"


def compare_variants(cli: argparse.Namespace) -> int:
    name_a, name_b = cli.compare
    try:
        before = read_peaks(cli.output / name_a / "control_peaks.csv")
        after = read_peaks(cli.output / name_b / "control_peaks.csv")
    except OSError as error:
        print(f"error: {error}")
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
    text = format_comparison(name_a, name_b, pairs, lost, new, skipped, cli.tolerance)
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
             "config.toml, which must hold one value)",
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
    return cli


def main(arguments: list[str] | None = None) -> int:
    cli = parse_cli_arguments(arguments)
    if cli.compare:
        return compare_variants(cli)
    return run_variant(cli)


if __name__ == "__main__":
    sys.exit(main())
