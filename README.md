# vibro

Weak-vibration analysis of building structures with an ADXL355
accelerometer (ESP32 firmware, binary `VIB2` packets over UART, Python
analysis).

## Workflow

Two commands produce a result you can show to someone:

```bash
python main.py --odr 250 --packets-per-session 8 --min-recommended-sessions 32
```

```bash
python stable_spectrum.py results/<capture>_raw.npz
```

The first records about 17 minutes and writes `results/<capture>_raw.npz`.
The second reads that file and writes `stable_results/<capture>/` with one
directory per spectral resolution (`nperseg_1024/`, `nperseg_2048/`,
`nperseg_4096/`), each holding the report, the peak table and the figures.
`figure_dominant_<capture>.png` is the one to show. `comparison.txt` and
`figure_compare_<capture>.png` next to them put the three resolutions side
by side.

The analysis figures of `main.py` are a different, older view of the same
data and can be skipped: its peak lists use fixed dB thresholds, which is
what made the picture change from run to run. Its raw `.npz` is what
matters. The recording checks that used to require looking at those figures
(sampling rate held, every axis alive, no knock dominating the record, no
sharp transient inside a packet) are now printed by `stable_spectrum.py`
under "Recording check".

Replay and the family analysis below stay useful for a different question:
when in time a structure was present, and whether it returns across separate
measurements.

## Measurement and replay

```bash
# physical capture (serial), saves .txt report, figures and raw .npz
python main.py --odr 250 --packets-per-session 8 --min-recommended-sessions 32

# offline multi-scale replay of a raw capture (no serial access)
python main.py --replay "results/<raw_file>.npz" --virtual-mode all
```

Replay writes `replay_results/<raw_stem>/` with one directory per virtual
layout (`4x4 ... 4x64`, `8x4 ... 8x32`) and per virtual run, plus
`replay_runs.csv`, `replay_regions.csv` and `replay_metadata.txt`.

On Windows, when stdout is redirected to a file, run with `PYTHONUTF8=1`
(the report contains `σ`).

## Frequency-family analysis (research layer)

`family_analysis.py` reads one or more replay result directories and groups
trusted regions from all temporal windows into recurring frequency
families. It is fully data-driven (no hard-coded building frequencies), it
does not modify the detector and it does not add a trusted gate.

```bash
# one capture
python family_analysis.py replay_results/<raw_stem>

# several independent physical captures (same ODR)
python family_analysis.py replay_results/<capture_1> replay_results/<capture_2> \
    --output family_results/<name>

# measurement reports without raw data can be mixed in: each .txt report
# enters as one window of its own layout (e.g. 8x8) and one capture
python family_analysis.py replay_results/<capture_1> results/<old_run>.txt ...

python -m unittest test_family_analysis
```

Method:

* observation = one trusted region in one virtual run;
* families are built per `(axis, band)`;
* distance `d = 0.5 * (|ΔFreq| + |ΔMed.Freq|)`, where `Freq` is the
  session-recurrence centre and `Med.Freq` the Median PSD maximum;
* two observations are incompatible when their session-peak ranges are
  separated by more than 1.5 Welch bins, or when `|ΔFreq|` or
  `|ΔMed.Freq|` exceeds `--max-family-span-hz` (default 2 × link tolerance);
* average-linkage clustering cut at `--link-tolerance-hz` (default: the
  effective ODR tolerance recorded by replay, 0.40 Hz at ODR 250);
  `--linkage complete` is stricter;
* one representative observation per family per virtual run (highest
  support fraction, then Median prominence, then closest `Freq`).

Outputs in `family_results/<name>/`:

| File | Content |
| --- | --- |
| `family_observations.csv` | every trusted region with its family id and representative flag |
| `family_summary.csv` | per family: Freq / Med.Freq centre and spread, range envelope, capture recurrence, median support / prominence / contrast, occupancy per layout |
| `family_scale_profile.csv` | per family × layout: windows, occupancy, centres, spreads |
| `family_capture_matrix.csv` | per family × physical capture: presence and per-layout occupancy |
| `family_pair_comparison.csv` | controlled same-packet pairs `4x8/8x4`, `4x16/8x8`, `4x32/8x16`, `4x64/8x32`: both / only-4 / only-8 counts and mean shifts of Freq, Med.Freq, support, prominence, contrast |
| `family_metadata.txt` | parameters, method, independence notes |
| `figure_family_time_<capture>.png` | frequency × time matrix per layout and axis |
| `figure_family_occupancy.png` | family × layout occupancy heat map |
| `figure_family_freq_vs_med.png` | Freq vs Med.Freq per observation |

Statistical caveat: nested layouts of one capture share packets and are not
independent samples. Occupancy is therefore reported separately per layout,
capture recurrence counts independent raw captures, and cross-scale
consistency is a separate profile rather than a pooled count.

## Stable spectrum (repeatable result from one capture)

`stable_spectrum.py` answers the question that has a repeatable answer:
which peaks of this capture are larger than the uncertainty of the spectral
estimate that produced them?

```bash
python stable_spectrum.py real_results/<capture>_raw.npz
python stable_spectrum.py real_results/*_raw.npz --output stable_results/real_all
python -m unittest test_stable_spectrum
```

Method: one periodogram per packet, mean power spectral density across
packets, per-bin standard error taken from the spread across packets, a
running-median baseline, and peak significance `z = prominence_dB /
standard_error_dB`. A peak is reported when `z` passes a Bonferroni-corrected
Student quantile for the number of bins tested, with `packets - 1` degrees of
freedom because the error bar is estimated from the same packets. Nothing is
tuned: the threshold follows from the bin count and the record length.

The report states the detection limit in dB, so "nothing rose above the
noise" comes with the number that would have been needed. That verdict is
itself a stable result, and it says a longer or better-excited recording is
required rather than a lower threshold.

Peaks are flagged `persistent` when they also clear the scaled threshold in
both of two interleaved halves of the record that share no samples. Below
64 packets (or 64 independent segments) the error bar is itself too noisy
and the report says so.

### Spectral resolution

Every capture is analysed at three segment lengths by default:

```bash
python stable_spectrum.py results/<capture>_raw.npz                  # 1024 2048 4096
python stable_spectrum.py results/<capture>_raw.npz --nperseg 1024 8192
```

| nperseg | Bin at ODR 250 | Segments from 256 packets |
| --- | --- | --- |
| 1024 | 0.25 Hz | 256, one per packet |
| 2048 | 0.12 Hz | 255, 50% overlap |
| 4096 | 0.06 Hz | 127, 50% overlap |

Up to one packet, a segment stays inside its packet. Longer segments are
cut from the packets joined into one continuous series. Their error bar is
corrected for the overlap, and the two halves used for `persistent` take
alternate non-overlapping segments. The support windows stay 16 packets
long at every resolution, so support counts are comparable.

What to look for when comparing the directories:

* a narrow line, such as a lightly damped mode, gets taller with finer bins,
  up to four times (6 dB) from 1024 to 4096, and its frequency is read to
  0.06 Hz instead of 0.25 Hz;
* a structure wider than the bin keeps its height, so finer bins only cost
  it error bar;
* the detection limit in dB rises with finer bins, from about 1.0 to 1.6 dB
  on a 256-packet record, because fewer segments are averaged and four
  times more bins are tested. A narrow line still wins because it grows
  faster than the limit.

Joining packets assumes none were lost. The recording check reports a step
at a packet join, which catches glitches, but the raw file stores no packet
sequence numbers, so a silently lost packet in a noise-dominated record
cannot be ruled out. Measured on pure noise, the share of analyses with a
false peak stays at the nominal 1% at every resolution.

Two figures are written per capture. `figure_dominant_<capture>.png` is the
one to show a reader: measured PSD per axis in g²/Hz, with everything the
record cannot distinguish from sensor noise shaded out. A curve that leaves
the shaded region is a real spectral structure, and it is labelled with its
frequency, its support over independent windows, and its prominence in dB.
`figure_stable_spectrum_<capture>.png` shows the same thing as prominence in
dB against the error band, which is easier to read when peaks are small.

Support is counted over consecutive non-overlapping windows of the record.
The frequency has already been chosen by the full-record estimate, so each
window is one pre-registered test rather than a search and an uncorrected
one-sided quantile applies; under pure noise a frequency would collect
support in about 5% of the windows.

To ask about a specific frequency, detected or not:

```bash
python stable_spectrum.py real_results/<capture>_raw.npz --probe 3.66 3.17
```

Each probe reports the prominence measured there, the prominence that would
have been needed, and the 95% upper bound. A structure stronger than that
bound is excluded by the record; a weaker one is not. This turns "we saw
nothing" into a measurement.

### Measuring repeatability

`repeatability_check.py` splits each capture into halves that share no
packets and compares the two answers:

```bash
python repeatability_check.py real_results/*_raw.npz --replay-root replay_results
```

`first/second` uses consecutive stretches of time and therefore measures how
stationary the building was; `even/odd` interleaves packets so both halves
see the same excitation, and a disagreement there is the estimator's own
instability. With `--replay-root` the same comparison runs for the existing
trusted-region detector as a baseline.

Comparisons where both halves found nothing are counted separately, because
repeating an empty answer costs the estimator nothing and averaging those in
inflates the score.

On the six evening captures in `real_results/`, 9 of 12 comparisons have both
halves empty. Of the 3 that carry a frequency, agreement is 0.33: one match
(13.65 Hz on the interleaved split), one genuine intermittency (the same
structure confined to the first half of that record), and one false peak at
32 packets, below the calibrated length. The existing detector scores 0.20,
but every one of its comparisons is substantive because it never returns an
empty answer, so the two numbers are not directly comparable.

These captures cannot test repeatability of a detection any further: they
contain almost nothing to detect.
