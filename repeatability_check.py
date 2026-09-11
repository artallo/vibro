"""Measure how repeatable a frequency result is, on the data itself.

A capture is split into two halves that contain no shared packets, each
half is analysed on its own, and the two answers are compared. This is the
closest thing to "would a second measurement give the same picture?" that
one recording can provide.

Two splits are used because they fail in different ways:

* ``first/second`` — two consecutive stretches of time. A structure that
  is only excited during part of the record shows up in one half only, so
  this split measures how stationary the building was.
* ``even/odd`` — interleaved packets. Both halves span the whole record
  and see the same excitation, so a disagreement here is the estimator's
  own instability rather than the building's.

When a replay directory is given for a capture, the same comparison is run
for the existing detector by taking the trusted regions it reported for
independent virtual runs. That gives the baseline the new estimator has to
beat.

Usage::

    python repeatability_check.py real_results/*_raw.npz
    python repeatability_check.py real_results/x_raw.npz --replay-root replay_results
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from stable_spectrum import (
    AXIS_KEYS,
    find_stable_peaks,
    analyze_axis,
    load_band_names,
    load_settings,
    significance_threshold,
)

MATCH_TOLERANCE_HZ = 0.40
BASELINE_LAYOUT = "8x4"


def analyse_subset(
    axes: dict[str, np.ndarray],
    sampling_rate_hz: float,
    packet_indices: np.ndarray,
    settings,
    bands,
) -> set[tuple[str, float]]:
    spectra = [
        analyze_axis(axis, axes[key][packet_indices], sampling_rate_hz, settings)
        for axis, key in AXIS_KEYS
    ]
    bins_tested = sum(spectrum.frequencies.size for spectrum in spectra)
    threshold = significance_threshold(
        bins_tested, settings.alpha, int(packet_indices.size),
    )
    peaks = find_stable_peaks(spectra, threshold, settings, bands)
    return {(peak.axis, round(peak.frequency_hz, 3)) for peak in peaks}


def agreement(first: set, second: set) -> float:
    """Share of the larger answer that the other answer also contains.

    Two empty answers score 1.0, but that agreement is trivial: repeating
    "nothing rose above the noise" costs the estimator nothing. Such
    comparisons are counted separately from the ones that carry a
    frequency, because averaging them together hides how often the
    non-empty answers actually match.
    """
    if not first and not second:
        return 1.0
    matched = sum(
        1
        for axis, frequency in first
        if any(
            axis == other_axis
            and abs(frequency - other_frequency) <= MATCH_TOLERANCE_HZ
            for other_axis, other_frequency in second
        )
    )
    return matched / max(len(first), len(second))


def describe(answer: set[tuple[str, float]]) -> str:
    if not answer:
        return "none"
    return " ".join(
        f"{axis}{frequency:.1f}" for axis, frequency in sorted(answer)
    )


def baseline_answers(
    replay_directory: Path,
    layout: str,
) -> tuple[set, set] | None:
    """Trusted regions of the existing detector, split into two halves."""
    regions_path = replay_directory / "replay_regions.csv"
    if not regions_path.exists():
        return None
    per_run: dict[int, set[tuple[str, float]]] = {}
    with regions_path.open(encoding="utf-8", newline="") as regions_file:
        for row in csv.DictReader(regions_file):
            if row["mode"] != layout:
                continue
            per_run.setdefault(int(row["virtual_run"]), set()).add(
                (row["axis"], round(float(row["med_freq_hz"]), 3))
            )
    if len(per_run) < 2:
        return None
    runs = sorted(per_run)
    middle = len(runs) // 2
    first = set().union(*(per_run[run] for run in runs[:middle]))
    second = set().union(*(per_run[run] for run in runs[middle:]))
    return first, second


def check_capture(
    raw_path: Path,
    replay_root: Path | None,
    settings,
    bands,
) -> list[dict]:
    with np.load(raw_path, allow_pickle=False) as archive:
        axes = {key: np.asarray(archive[key]) for _, key in AXIS_KEYS}
        packet_fs_hz = np.asarray(archive["packet_fs_hz"])
    sampling_rate_hz = float(np.mean(packet_fs_hz))
    packet_count = axes["x"].shape[0]
    splits = {
        "first/second": (
            np.arange(0, packet_count // 2),
            np.arange(packet_count // 2, packet_count),
        ),
        "even/odd": (
            np.arange(0, packet_count, 2),
            np.arange(1, packet_count, 2),
        ),
    }
    rows = []
    for split_name, (first_indices, second_indices) in splits.items():
        first = analyse_subset(
            axes, sampling_rate_hz, first_indices, settings, bands,
        )
        second = analyse_subset(
            axes, sampling_rate_hz, second_indices, settings, bands,
        )
        rows.append({
            "capture": raw_path.stem,
            "packets": packet_count,
            "split": split_name,
            "agreement": agreement(first, second),
            "first": describe(first),
            "second": describe(second),
            "baseline_agreement": None,
        })
    if replay_root is not None:
        baseline = baseline_answers(
            replay_root / raw_path.stem, BASELINE_LAYOUT,
        )
        if baseline is not None:
            rows[0]["baseline_agreement"] = agreement(*baseline)
    return rows


def print_table(rows: list[dict]) -> None:
    header = (
        f"{'capture':<34}{'pkts':>5}  {'split':<13}"
        f"{'existing':>9}{'stable':>8}   answers"
    )
    print(header)
    print("-" * (len(header) + 24))
    for row in rows:
        baseline = (
            f"{row['baseline_agreement']:.2f}"
            if row["baseline_agreement"] is not None else ""
        )
        print(
            f"{row['capture'][:33]:<34}{row['packets']:>5}  "
            f"{row['split']:<13}{baseline:>9}{row['agreement']:>8.2f}   "
            f"{row['first']} | {row['second']}"
        )
    empty = [
        row for row in rows
        if row["first"] == "none" and row["second"] == "none"
    ]
    substantive = [row for row in rows if row not in empty]
    baselines = [
        row["baseline_agreement"] for row in rows
        if row["baseline_agreement"] is not None
    ]
    print("-" * (len(header) + 24))
    print(
        f"{len(rows)} comparisons: {len(empty)} where both halves found "
        f"nothing (agreement is trivial), {len(substantive)} carrying a "
        "frequency"
    )
    if substantive:
        print(
            "agreement where at least one half found something: "
            f"{np.mean([row['agreement'] for row in substantive]):.2f}"
        )
    else:
        print(
            "no comparison produced a frequency, so repeatability of a "
            "detection is untested here"
        )
    if baselines:
        print(
            f"existing detector, all comparisons substantive because it "
            f"never returns an empty answer: {np.mean(baselines):.2f}"
        )


def parse_cli_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure repeatability on disjoint halves of a capture",
    )
    parser.add_argument("raw_paths", nargs="+", type=Path)
    parser.add_argument(
        "--replay-root", type=Path, default=None,
        help="directory holding replay results, to compare against the "
             "existing detector",
    )
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--csv", type=Path, default=None)
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    cli = parse_cli_arguments(arguments)
    settings = load_settings(cli.alpha, None)
    bands = load_band_names()
    rows: list[dict] = []
    for raw_path in cli.raw_paths:
        rows.extend(check_capture(raw_path, cli.replay_root, settings, bands))
    print_table(rows)
    if cli.csv is not None:
        cli.csv.parent.mkdir(parents=True, exist_ok=True)
        with cli.csv.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved: {cli.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
