import struct
from dataclasses import dataclass
from typing import Any

import serial
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.signal import find_peaks

# ==========================================================
# Настройки
# ==========================================================

PORT = "COM8"
BAUD = 115200

# AXIS = "X"          # X / Y / Z

MAGIC = b"VIB2"

# Welch
NPERSEG = 1024
NOVERLAP = 512
PACKETS_PER_SESSION = 8
MIN_RECOMMENDED_SESSIONS = 5

# Диапазоны анализа
PSD_MIN_FREQ = 0.5
PSD_MAX_FREQ = 20
ANALYSIS_BANDS = [
    ("Full", PSD_MIN_FREQ, PSD_MAX_FREQ),
]

# ----------------------------------------------------------
# Peak detection
# ----------------------------------------------------------

PROMINENCE = 0.3          # 30% от максимума
MIN_DISTANCE_HZ = 1.0
MIN_STABILITY = 5.0

COLORS = {
    "X": "tab:blue",
    "Y": "tab:orange",
    "Z": "tab:green",
}

# ==========================================================


@dataclass
class AnalysisBand:
    name: str
    min_frequency: float
    max_frequency: float


def build_analysis_bands() -> list[AnalysisBand]:
    if not ANALYSIS_BANDS:
        raise ValueError("At least one analysis band is required")

    bands = []
    for entry in ANALYSIS_BANDS:
        if not isinstance(entry, (tuple, list)) or len(entry) != 3:
            raise ValueError(
                "Each analysis band must contain name, min_frequency, and max_frequency"
            )

        name, min_frequency, max_frequency = entry
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Analysis band name must be a non-empty string")

        try:
            min_frequency = float(min_frequency)
            max_frequency = float(max_frequency)
        except (TypeError, ValueError):
            raise ValueError("Analysis band frequencies must be numeric") from None

        if min_frequency < 0:
            raise ValueError("Analysis band minimum frequency must be non-negative")
        if min_frequency >= max_frequency:
            raise ValueError(
                "Analysis band minimum frequency must be less than maximum frequency"
            )
        if max_frequency > PSD_MAX_FREQ:
            raise ValueError(
                "Analysis band maximum frequency must not exceed PSD_MAX_FREQ"
            )

        bands.append(
            AnalysisBand(
                name=name.strip(),
                min_frequency=min_frequency,
                max_frequency=max_frequency,
            )
        )

    return bands


@dataclass
class FFTResult:
    freq: np.ndarray
    amplitude: np.ndarray
    resolution: float


@dataclass
class AverageFFTResult:
    freq: np.ndarray
    amplitude: np.ndarray


@dataclass
class PSDResult:
    freq: np.ndarray
    psd: np.ndarray
    resolution: float


@dataclass
class AxisResult:
    fft: FFTResult
    average_fft: AverageFFTResult
    psd: PSDResult


@dataclass
class SessionResult:
    number: int
    fs: float
    duration: float
    samples: int
    x: AxisResult
    y: AxisResult
    z: AxisResult


@dataclass
class AxisStatistics:
    median: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    stability: np.ndarray


@dataclass
class StatisticsResult:
    x: AxisStatistics
    y: AxisStatistics
    z: AxisStatistics


@dataclass
class AxisPeaks:
    frequencies: np.ndarray
    amplitudes: np.ndarray
    properties: dict[str, Any]


@dataclass
class PeakResult:
    x: AxisPeaks
    y: AxisPeaks
    z: AxisPeaks


@dataclass
class VisualizationAxis:
    frequency: np.ndarray
    median_psd: np.ndarray
    mean_psd: np.ndarray
    std_psd: np.ndarray
    stability: np.ndarray
    peak_frequencies: np.ndarray
    peak_amplitudes: np.ndarray


@dataclass
class VisualizationData:
    x: VisualizationAxis
    y: VisualizationAxis
    z: VisualizationAxis


def build_visualization_data(
    statistics: StatisticsResult,
    peaks: PeakResult,
    sessions: list[SessionResult],
) -> VisualizationData:
    if not sessions:
        raise ValueError("At least one completed session is required")

    session = sessions[0]
    frequency = session.x.psd.freq

    if not np.array_equal(frequency, session.y.psd.freq):
        raise ValueError("PSD frequency axes do not match")
    if not np.array_equal(frequency, session.z.psd.freq):
        raise ValueError("PSD frequency axes do not match")

    def build_axis(
        axis_name: str,
        axis_statistics: AxisStatistics,
        axis_peaks: AxisPeaks,
    ) -> VisualizationAxis:
        expected_length = len(frequency)
        arrays = {
            "median": axis_statistics.median,
            "mean": axis_statistics.mean,
            "std": axis_statistics.std,
            "stability": axis_statistics.stability,
        }

        for array_name, array in arrays.items():
            if len(array) != expected_length:
                raise ValueError(
                    f"Frequency axis length does not match {axis_name} {array_name} PSD length"
                )

        if len(axis_peaks.frequencies) != len(axis_peaks.amplitudes):
            raise ValueError(
                f"Peak frequency and amplitude lengths do not match for {axis_name}"
            )

        return VisualizationAxis(
            frequency=frequency,
            median_psd=axis_statistics.median,
            mean_psd=axis_statistics.mean,
            std_psd=axis_statistics.std,
            stability=axis_statistics.stability,
            peak_frequencies=axis_peaks.frequencies,
            peak_amplitudes=axis_peaks.amplitudes,
        )

    return VisualizationData(
        x=build_axis("X", statistics.x, peaks.x),
        y=build_axis("Y", statistics.y, peaks.y),
        z=build_axis("Z", statistics.z, peaks.z),
    )


def find_psd_peaks(
    statistics: StatisticsResult,
    sessions: list[SessionResult],
) -> PeakResult:
    if not sessions:
        raise ValueError("At least one completed session is required")

    session = sessions[0]
    freq = session.x.psd.freq

    if not np.array_equal(freq, session.y.psd.freq):
        raise ValueError("PSD frequency axes do not match")
    if not np.array_equal(freq, session.z.psd.freq):
        raise ValueError("PSD frequency axes do not match")

    if len(freq) != len(statistics.x.median):
        raise ValueError("Frequency axis length does not match X median PSD length")
    if len(freq) != len(statistics.y.median):
        raise ValueError("Frequency axis length does not match Y median PSD length")
    if len(freq) != len(statistics.z.median):
        raise ValueError("Frequency axis length does not match Z median PSD length")

    def detect_axis_peaks(
        median_psd: np.ndarray,
        stability: np.ndarray,
        freq: np.ndarray,
    ) -> AxisPeaks:
        if len(median_psd) == 0:
            return AxisPeaks(
                frequencies=np.array([]),
                amplitudes=np.array([]),
                properties={}
            )

        if len(freq) < 2:
            raise ValueError("Frequency axis must contain at least two values")

        if len(stability) != len(median_psd):
            raise ValueError("Stability length does not match median PSD length")

        prominence = np.max(median_psd) * PROMINENCE
        resolution = freq[1] - freq[0]
        distance = max(int(MIN_DISTANCE_HZ / resolution), 1)

        peak_indices, properties = find_peaks(
            median_psd,
            prominence=prominence,
            distance=distance,
        )

        stable_peak_mask = stability[peak_indices] >= MIN_STABILITY
        peak_indices = peak_indices[stable_peak_mask]
        properties = {
            name: values[stable_peak_mask]
            for name, values in properties.items()
        }

        return AxisPeaks(
            frequencies=freq[peak_indices],
            amplitudes=median_psd[peak_indices],
            properties=properties,
        )

    return PeakResult(
        x=detect_axis_peaks(statistics.x.median, statistics.x.stability, freq),
        y=detect_axis_peaks(statistics.y.median, statistics.y.stability, freq),
        z=detect_axis_peaks(statistics.z.median, statistics.z.stability, freq),
    )


def compute_statistics(
    sessions: list[SessionResult],
) -> StatisticsResult:
    def compute_axis_statistics(axis: str) -> AxisStatistics:
        psd_stack = np.stack(
            [getattr(session, axis).psd.psd for session in sessions],
            axis=0
        )

        median = np.median(psd_stack, axis=0)
        mean = np.mean(psd_stack, axis=0)
        std = np.std(psd_stack, axis=0)
        stability = np.divide(
            mean,
            std,
            out=np.zeros_like(mean),
            where=std != 0
        )

        return AxisStatistics(
            median=median,
            mean=mean,
            std=std,
            stability=stability
        )

    return StatisticsResult(
        x=compute_axis_statistics("x"),
        y=compute_axis_statistics("y"),
        z=compute_axis_statistics("z")
    )


# ==========================================================

analysis_bands = build_analysis_bands()

ser = serial.Serial(PORT, BAUD, timeout=2)

print("Connected:", PORT)


# ==========================================================
# Поиск начала пакета
# ==========================================================

def find_magic():

    while True:

        b = ser.read(1)

        if not b:
            continue

        if b == MAGIC[:1]:

            rest = ser.read(3)

            if b + rest == MAGIC:
                return


# ==========================================================

def read_packet():

    find_magic()

    header = ser.read(12)

    if len(header) != 12:
        return None

    # N, elapsed = struct.unpack("<HI", header)
    N, _, elapsed, sequence = struct.unpack("<HHII", header)

    data = ser.read(N * 12)

    if len(data) != N * 12:
        return None

    values = np.frombuffer(data, dtype="<i4").astype(np.float64)

    values /= 256000.0

    x = values[0:N]
    y = values[N:2 * N]
    z = values[2 * N:3 * N]

    fs = (N - 1) / elapsed * 1e6

    return fs, x, y, z


# ==========================================================
# FFT
# ==========================================================

def compute_fft(signal, fs):

    window = np.hanning(len(signal))

    fft = np.fft.rfft(signal * window)

    freq = np.fft.rfftfreq(len(signal), 1 / fs)

    amplitude = np.abs(fft)

    amplitude *= 2.0 / np.sum(window)

    resolution = fs / len(signal)

    return FFTResult(
        freq=freq,
        amplitude=amplitude,
        resolution=resolution
    )

def compute_average_fft(signal, fs):

    nperseg = min(NPERSEG, len(signal))
    noverlap = min(NOVERLAP, nperseg // 2)
    step = nperseg - noverlap

    window = np.hanning(nperseg)

    spectra = []

    for start in range(0, len(signal) - nperseg + 1, step):

        segment = signal[start:start + nperseg]

        fft = np.fft.rfft(segment * window)

        amp = np.abs(fft)

        amp *= 2.0 / np.sum(window)

        spectra.append(amp)

    amplitude = np.mean(spectra, axis=0)

    freq = np.fft.rfftfreq(nperseg, 1 / fs)

    return AverageFFTResult(
        freq=freq,
        amplitude=amplitude
    )

# ==========================================================
# Welch PSD
# ==========================================================

def compute_psd(signal, fs):

    freq, psd = welch(
        signal,
        fs=fs,
        window="hann",
        nperseg=min(NPERSEG, len(signal)),
        noverlap=min(NOVERLAP, len(signal)//2),
        scaling="density"
    )

    mask = (freq >= PSD_MIN_FREQ) & (freq <= PSD_MAX_FREQ)

    freq = freq[mask]
    psd = psd[mask]

    resolution = fs / min(NPERSEG, len(signal))

    return PSDResult(
        freq=freq,
        psd=psd,
        resolution=resolution
    )

def process_session(session, session_fs, number):

    if any(len(session[axis]) != PACKETS_PER_SESSION for axis in ("X", "Y", "Z")):
        raise ValueError("Cannot process incomplete session")

    if len(session_fs) != PACKETS_PER_SESSION:
        raise ValueError("Cannot process incomplete session")

    fs = np.mean(session_fs)

    axis_results = {}

    for axis in ("X", "Y", "Z"):

        signal = np.concatenate(session[axis])

        signal = signal - np.mean(signal)

        axis_results[axis] = AxisResult(
            fft=compute_fft(signal, fs),
            average_fft=compute_average_fft(signal, fs),
            psd=compute_psd(signal, fs)
        )

    samples = len(np.concatenate(session["X"]))

    duration = samples / fs

    return SessionResult(
        number=number,
        fs=fs,
        duration=duration,
        samples=samples,
        x=axis_results["X"],
        y=axis_results["Y"],
        z=axis_results["Z"]
    )


# ==========================================================

print()
input("Press ENTER to start recording...")
ser.reset_input_buffer()

print()
print("Recording...")
print("Press Ctrl+C to stop.")
print()

current_session = {
    "X": [],
    "Y": [],
    "Z": [],
}
current_session_fs = []
sessions = []
stop_requested = False
fs_list = []

while True:

    try:
        packet = read_packet()
    except KeyboardInterrupt:
        stop_requested = True
        if len(current_session["X"]) == 0:
            break
        continue

    if packet is None:
        continue

    fs, x, y, z = packet

    current_session["X"].append(x)
    current_session["Y"].append(y)
    current_session["Z"].append(z)
    current_session_fs.append(fs)

    fs_list.append(fs)

    session_packets = len(current_session["X"])

    if session_packets == PACKETS_PER_SESSION:
        session_result = process_session(
            current_session,
            current_session_fs,
            len(sessions) + 1,
        )
        sessions.append(session_result)
        current_session = {
            "X": [],
            "Y": [],
            "Z": [],
        }
        current_session_fs = []
    
    packets = len(fs_list)
    print(
        f"\rPackets: {packets:4d}"
        f"   Duration: {packets * len(x) / np.mean(fs_list):6.1f} s"
        f"   Fs={np.mean(fs_list):6.2f}",
        end=""
    )

    if stop_requested and session_packets == PACKETS_PER_SESSION:
        break

print()

statistics: StatisticsResult | None = None
peaks: PeakResult | None = None
visualization_data: VisualizationData | None = None

if sessions:
    statistics = compute_statistics(sessions)
    peaks = find_psd_peaks(statistics, sessions)
    visualization_data = build_visualization_data(
        statistics,
        peaks,
        sessions,
    )

if not sessions:
    print("No completed sessions available for analysis.")
    ser.close()
    raise SystemExit(0)

if len(sessions) < MIN_RECOMMENDED_SESSIONS:
    print(
        f"Warning: only {len(sessions)} completed session(s); "
        f"at least {MIN_RECOMMENDED_SESSIONS} are recommended."
    )

# ==========================================================
# Statistical PSD visualization
# ==========================================================

if visualization_data is None:
    raise RuntimeError("Visualization data was not built")

stat_fig, stat_axes = plt.subplots(
    6,
    1,
    figsize=(14, 16),
    sharex=True,
)

visualization_axes = {
    "X": visualization_data.x,
    "Y": visualization_data.y,
    "Z": visualization_data.z,
}

for axis_index, (axis_name, axis_data) in enumerate(visualization_axes.items()):
    psd_axis = stat_axes[axis_index * 2]
    stability_axis = stat_axes[axis_index * 2 + 1]

    psd_axis.semilogy(
        axis_data.frequency,
        axis_data.median_psd,
        color=COLORS[axis_name],
        label=f"{axis_name} Median PSD",
    )
    psd_axis.scatter(
        axis_data.peak_frequencies,
        axis_data.peak_amplitudes,
        color=COLORS[axis_name],
        marker="x",
        label="Stable peaks",
    )
    psd_axis.set_title(f"Axis {axis_name} — Median PSD")
    psd_axis.set_ylabel("PSD [g²/Hz]")
    psd_axis.set_xlim(PSD_MIN_FREQ, PSD_MAX_FREQ)
    psd_axis.grid(True, which="major", alpha=0.6)
    psd_axis.grid(True, which="minor", alpha=0.2)
    psd_axis.legend()

    peak_stability = np.interp(
        axis_data.peak_frequencies,
        axis_data.frequency,
        axis_data.stability,
    )
    stability_axis.plot(
        axis_data.frequency,
        axis_data.stability,
        color=COLORS[axis_name],
        label="Stability",
    )
    stability_axis.axhline(
        MIN_STABILITY,
        linestyle="--",
        color="black",
        label="Threshold",
    )
    stability_axis.scatter(
        axis_data.peak_frequencies,
        peak_stability,
        color=COLORS[axis_name],
        marker="x",
        label="Stable peaks",
    )
    stability_axis.set_title(f"Axis {axis_name} — Stability")
    stability_axis.set_ylabel("Mean / Std")
    stability_axis.set_xlim(PSD_MIN_FREQ, PSD_MAX_FREQ)
    stability_axis.grid(True, which="major", alpha=0.6)
    stability_axis.legend()

stat_axes[-1].set_xlabel("Frequency [Hz]")
stat_fig.suptitle("Statistical PSD, stability, and stable peaks")
stat_fig.tight_layout(rect=(0, 0, 1, 0.97))


plt.show()
