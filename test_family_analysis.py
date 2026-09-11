"""Unit tests for the family-analysis clustering rules.

Run with: python -m pytest test_family_analysis.py  (or python test_family_analysis.py)
"""

from __future__ import annotations

import unittest

from family_analysis import Observation, build_families

TOLERANCE = 0.40
SPAN = 0.80
PADDING = 0.75 * 250.0 / 1024


def make_observation(
    observation_id: int,
    freq: float,
    med: float,
    range_min: float,
    range_max: float,
    *,
    axis: str = "X",
    mode: str = "4x4",
    virtual_run: int = 1,
    capture: str = "cap",
    support: float = 0.75,
    prom: float = 2.0,
) -> Observation:
    return Observation(
        observation_id=observation_id,
        capture=capture,
        source=f"{capture}.npz",
        odr_hz=250.0,
        frequency_tolerance_hz=TOLERANCE,
        mode=mode,
        packets_per_session=4,
        sessions_per_run=4,
        virtual_run=virtual_run,
        packet_start=1 + 16 * (virtual_run - 1),
        packet_end=16 * virtual_run,
        duration_seconds=65.0,
        axis=axis,
        band="Low frequency",
        freq_hz=freq,
        med_freq_hz=med,
        support_n=3,
        support_total=4,
        support_fraction=support,
        range_min_hz=range_min,
        range_max_hz=range_max,
        frequency_std_hz=0.1,
        med_prom_db=prom,
        med_contrast_db=3.0,
        band_contrast_db=1.5,
        sources=1,
        weight=1.0,
    )


class FamilyClusteringTests(unittest.TestCase):
    def test_no_chaining_across_wide_span(self) -> None:
        # 3.2 -> 3.5 -> 3.8 -> 4.1: neighbours are within tolerance but the
        # ends are not; complete linkage must not merge everything.
        observations = [
            make_observation(1, 3.2, 3.2, 3.0, 3.4, virtual_run=1),
            make_observation(2, 3.5, 3.5, 3.3, 3.7, virtual_run=2),
            make_observation(3, 3.8, 3.8, 3.6, 4.0, virtual_run=3),
            make_observation(4, 4.1, 4.1, 3.9, 4.3, virtual_run=4),
        ]
        for linkage_method in ("complete", "average"):
            families = build_families(
                observations, TOLERANCE, SPAN, PADDING, linkage_method,
            )
            self.assertGreaterEqual(len(families), 2, linkage_method)
            for family in families:
                freqs = [o.freq_hz for o in family.observations]
                self.assertLessEqual(max(freqs) - min(freqs), SPAN)

    def test_adjacent_bin_ranges_stay_compatible(self) -> None:
        # Med.Freq quantised to neighbouring Welch bins (7.81 / 8.06 Hz at
        # ODR 250) with single-bin ranges must not be split by the range gate.
        observations = [
            make_observation(1, 7.81, 7.81, 7.81, 7.81, virtual_run=1),
            make_observation(2, 8.06, 8.06, 8.06, 8.30, virtual_run=2),
        ]
        families = build_families(observations, TOLERANCE, SPAN, PADDING)
        self.assertEqual(len(families), 1)

    def test_shifted_freq_with_same_spectral_maximum_merges(self) -> None:
        # Freq 3.41 / Med.Freq 3.66 and Freq 3.75 / Med.Freq 3.66 describe
        # the same spectral maximum and must land in one family.
        observations = [
            make_observation(1, 3.41, 3.66, 2.92, 3.90, virtual_run=1),
            make_observation(2, 3.75, 3.66, 3.50, 3.90, virtual_run=2),
        ]
        families = build_families(observations, TOLERANCE, SPAN, PADDING)
        self.assertEqual(len(families), 1)

    def test_disjoint_ranges_are_incompatible(self) -> None:
        observations = [
            # Freq/Med.Freq distance (0.375 Hz) would pass the link cut, but
            # the session-peak ranges are separated by 0.40 Hz, more than
            # 1.5 Welch bins at ODR 250 (0.366 Hz).
            make_observation(1, 3.40, 3.40, 3.20, 3.40, virtual_run=1),
            make_observation(2, 3.75, 3.80, 3.80, 3.95, virtual_run=2),
        ]
        families = build_families(observations, TOLERANCE, SPAN, PADDING)
        self.assertEqual(len(families), 2)

    def test_separate_axes_never_merge(self) -> None:
        observations = [
            make_observation(1, 3.2, 3.2, 3.0, 3.4, axis="X"),
            make_observation(2, 3.2, 3.2, 3.0, 3.4, axis="Y"),
        ]
        families = build_families(observations, TOLERANCE, SPAN, PADDING)
        self.assertEqual(len(families), 2)
        self.assertEqual(sorted(f.family_id for f in families),
                         ["F-X-01", "F-Y-01"])

    def test_one_representative_per_virtual_run(self) -> None:
        observations = [
            make_observation(1, 4.60, 4.65, 4.4, 4.8, virtual_run=1,
                             support=0.5, prom=1.6),
            make_observation(2, 4.80, 4.70, 4.6, 4.9, virtual_run=1,
                             support=0.75, prom=2.5),
            make_observation(3, 4.70, 4.68, 4.5, 4.9, virtual_run=2),
        ]
        families = build_families(observations, TOLERANCE, SPAN, PADDING)
        self.assertEqual(len(families), 1)
        family = families[0]
        self.assertEqual(len(family.observations), 3)
        self.assertEqual(len(family.representatives), 2)
        representative_ids = {o.observation_id for o in family.representatives}
        self.assertEqual(representative_ids, {2, 3})
        by_id = {o.observation_id: o for o in family.observations}
        self.assertEqual(by_id[1].regions_in_window, 2)
        self.assertEqual(by_id[3].regions_in_window, 1)

    def test_family_ids_ordered_by_frequency(self) -> None:
        observations = [
            make_observation(1, 9.2, 9.2, 9.0, 9.4, virtual_run=1),
            make_observation(2, 4.7, 4.7, 4.5, 4.9, virtual_run=1),
        ]
        families = build_families(observations, TOLERANCE, SPAN, PADDING)
        self.assertEqual([f.family_id for f in families], ["F-X-01", "F-X-02"])
        self.assertAlmostEqual(families[0].observations[0].freq_hz, 4.7)


SAMPLE_REPORT = """Run started: 2026-08-31 14:36:29
Run: 1/3
ODR: 250 Hz
Packets/session: 8
Target sessions: 8
Welch nperseg: 1024
Welch noverlap: 512

Packets:   64   Duration:  262.6 s   Fs=249.53

Session frequency clusters — X
Band            Freq Hz  Support  Support %   σf Hz       Range Hz  Med.Freq  Med.Prom  Med.Contr  Band.Contr  Med.Pass  Sessions
Low frequency      3.66    6/8         75.0    0.11      3.41–3.66      3.66      3.25       1.52        2.45      PASS  1,4,5,6,7,8

Trusted frequency regions — X
Band           Freq Hz  Support  Med.Freq  Med.Prom  Med.Contr  Band.Contr  Weight       Range Hz
Low frequency     3.66    6/8        3.66      3.25       1.52        2.45    1.00  3.41–3.66

Trusted frequency regions — Y
Band           Freq Hz  Support  Med.Freq  Med.Prom  Med.Contr  Band.Contr  Weight       Range Hz
Low frequency     3.17    8/8        3.17      8.20       6.22        7.51    1.00  2.92–3.41
High frequency   13.16    4/8       13.16      1.64       0.89        0.88    0.70  12.92–13.40

Trusted frequency regions — Z
Band           Freq Hz  Support  Med.Freq  Med.Prom  Med.Contr  Band.Contr  Weight       Range Hz
"""


class ReportParsingTests(unittest.TestCase):
    def test_parse_report_header_and_trusted_rows(self) -> None:
        from family_analysis import parse_report

        header, rows = parse_report(SAMPLE_REPORT)
        self.assertEqual(header["odr_hz"], 250.0)
        self.assertEqual(header["packets_per_session"], 8)
        self.assertEqual(header["target_sessions"], 8)
        self.assertEqual(header["packet_count"], 64)
        self.assertAlmostEqual(header["duration_seconds"], 262.6)
        self.assertNotIn("frequency_tolerance_hz", header)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["axis"], "X")
        self.assertEqual(rows[0]["band"], "Low frequency")
        self.assertAlmostEqual(rows[0]["freq_hz"], 3.66)
        self.assertEqual((rows[0]["support_n"], rows[0]["support_total"]),
                         (6, 8))
        self.assertAlmostEqual(rows[0]["range_max_hz"], 3.66)
        self.assertEqual(rows[2]["axis"], "Y")
        self.assertEqual(rows[2]["band"], "High frequency")
        self.assertAlmostEqual(rows[2]["weight"], 0.70)

    def test_load_report_builds_window_and_observations(self) -> None:
        import tempfile
        from pathlib import Path
        from family_analysis import load_report

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "20260831_143629_ODR250_run01.txt"
            path.write_text(SAMPLE_REPORT, encoding="utf-8")
            window, observations = load_report(path, 10, 0.40)
        self.assertEqual(window.mode, "8x8")
        self.assertEqual(window.capture, "20260831_143629_ODR250_run01")
        self.assertEqual((window.packet_start, window.packet_end), (1, 64))
        self.assertEqual([o.observation_id for o in observations],
                         [10, 11, 12])
        self.assertAlmostEqual(observations[0].support_fraction, 0.75)
        self.assertAlmostEqual(observations[0].frequency_tolerance_hz, 0.40)
        self.assertTrue(observations[0].frequency_std_hz != observations[0].frequency_std_hz)


if __name__ == "__main__":
    unittest.main()
