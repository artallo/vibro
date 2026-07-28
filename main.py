import struct
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import serial
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.signal import find_peaks

# ==========================================================
# РќР°СЃС‚СЂРѕР№РєРё
# ==========================================================

CONFIG_PATH = Path(__file__).with_name("config.toml")

# AXIS = "X"          # X / Y / Z

MAGIC = b"VIB2"

COLORS = {
    "X": "tab:blue",
    "Y": "tab:orange",
    "Z": "tab:green",
}

# ==========================================================


@dataclass(frozen=True)
class AnalysisBand:
    name: str
    min_frequency: float
    max_frequency: float
    prominence_db: float
    min_distance_hz: float
    min_stability: float


@dataclass(frozen=True)
class SerialConfig:
    port: str
    baud: int
    timeout_seconds: float


@dataclass(frozen=True)
class SessionConfig:
    packets_per_session: int
    min_recommended_sessions: int


@dataclass(frozen=True)
class WelchConfig:
    nperseg: int
    noverlap: int


@dataclass(frozen=True)
class ApplicationConfig:
    serial: SerialConfig
    session: SessionConfig
    welch: WelchConfig
    analysis_bands: list[AnalysisBand]


def validate_config(config: ApplicationConfig) -> None:
    def require_positive_integer(value: Any, name: str) -> None:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    def require_non_negative_integer(value: Any, name: str) -> None:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")

    def require_finite_number(value: Any, name: str) -> None:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not np.isfinite(value)
        ):
            raise ValueError(f"{name} must be a finite number")

    if not isinstance(config.serial.port, str) or not config.serial.port.strip():
        raise ValueError("Serial port must be a non-empty string")
    require_positive_integer(config.serial.baud, "Serial baud")
    require_finite_number(
        config.serial.timeout_seconds,
        "Serial timeout_seconds",
    )
    if config.serial.timeout_seconds < 0:
        raise ValueError("Serial timeout_seconds must be non-negative")

    require_positive_integer(
        config.session.packets_per_session,
        "Session packets_per_session",
    )
    require_positive_integer(
        config.session.min_recommended_sessions,
        "Session min_recommended_sessions",
    )

    require_positive_integer(config.welch.nperseg, "Welch nperseg")
    require_non_negative_integer(config.welch.noverlap, "Welch noverlap")
    if config.welch.noverlap >= config.welch.nperseg:
        raise ValueError("Welch noverlap must be less than nperseg")

    if not isinstance(config.analysis_bands, list) or not config.analysis_bands:
        raise ValueError("At least one analysis band is required")

    for band in config.analysis_bands:
        if not isinstance(band.name, str) or not band.name.strip():
            raise ValueError("Analysis band name must be a non-empty string")

        band_name = f"Analysis band {band.name!r}"
        require_finite_number(
            band.min_frequency,
            f"{band_name} minimum frequency",
        )
        require_finite_number(
            band.max_frequency,
            f"{band_name} maximum frequency",
        )
        require_finite_number(
            band.prominence_db,
            f"{band_name} prominence_db",
        )
        require_finite_number(
            band.min_distance_hz,
            f"{band_name} min_distance_hz",
        )
        require_finite_number(
            band.min_stability,
            f"{band_name} min_stability",
        )

        if band.min_frequency < 0:
            raise ValueError(
                f"{band_name} minimum frequency must be non-negative"
            )
        if band.max_frequency <= band.min_frequency:
            raise ValueError(
                f"{band_name} maximum frequency "
                "must be greater than minimum frequency"
            )
        if band.prominence_db < 0:
            raise ValueError(f"{band_name} prominence_db must be non-negative")
        if band.min_distance_hz < 0:
            raise ValueError(
                f"{band_name} min_distance_hz must be non-negative"
            )
        if band.min_stability < 0:
            raise ValueError(
                f"{band_name} min_stability must be non-negative"
            )


def load_config(path: Path) -> ApplicationConfig:
    with path.open("rb") as file:
        raw_config = tomllib.load(file)

    serial_data = raw_config["serial"]
    session_data = raw_config["session"]
    welch_data = raw_config["welch"]
    band_entries = raw_config["analysis"]["bands"]

    analysis_bands = [
        AnalysisBand(
            name=entry["name"],
            min_frequency=entry["min_frequency"],
            max_frequency=entry["max_frequency"],
            prominence_db=entry["prominence_db"],
            min_distance_hz=entry["min_distance_hz"],
            min_stability=entry["min_stability"],
        )
        for entry in band_entries
    ]

    config = ApplicationConfig(
        serial=SerialConfig(
            port=serial_data["port"],
            baud=serial_data["baud"],
            timeout_seconds=serial_data["timeout_seconds"],
        ),
        session=SessionConfig(
            packets_per_session=session_data["packets_per_session"],
            min_recommended_sessions=session_data["min_recommended_sessions"],
        ),
        welch=WelchConfig(
            nperseg=welch_data["nperseg"],
            noverlap=welch_data["noverlap"],
        ),
        analysis_bands=analysis_bands,
    )
    validate_config(config)
    return config


def get_analysis_frequency_limits(
    analysis_bands: list[AnalysisBand],
) -> tuple[float, float]:
    return (
        min(band.min_frequency for band in analysis_bands),
        max(band.max_frequency for band in analysis_bands),
    )


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


def draw_analysis_bands(
    ax,
    analysis_bands: list[AnalysisBand],
) -> None:
    boundaries = sorted({
        boundary
        for band in analysis_bands
        for boundary in (band.min_frequency, band.max_frequency)
    })

    for boundary in boundaries:
        ax.axvline(
            boundary,
            linestyle="--",
            linewidth=0.8,
            alpha=0.5,
        )

    for band in analysis_bands:
        label_frequency = (band.min_frequency + band.max_frequency) / 2
        ax.text(
            label_frequency,
            0.97,
            band.name,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize="small",
            alpha=0.7,
        )


def annotate_peak_frequencies(
    ax,
    peak_frequencies,
    peak_values,
) -> None:
    if len(peak_frequencies) != len(peak_values):
        raise ValueError("Peak frequency and value arrays must have equal lengths")

    for index in range(len(peak_frequencies)):
        frequency = peak_frequencies[index]
        value = peak_values[index]
        ax.annotate(
            f"{frequency:.1f} Hz",
            xy=(frequency, value),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


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


def _find_axis_peaks_by_bands(
    freq: np.ndarray,
    median: np.ndarray,
    stability: np.ndarray,
    analysis_bands: list[AnalysisBand],
) -> AxisPeaks:
    if len(median) != len(freq):
        raise ValueError("Frequency axis length does not match median PSD length")

    if not np.all(np.isfinite(median)):
        raise ValueError("Median PSD must contain only finite values")

    if np.any(median < 0):
        raise ValueError("Median PSD must not contain negative values")

    for band in analysis_bands:
        if band.prominence_db < 0:
            raise ValueError(
                "Analysis band prominence_db must be non-negative"
            )
        if band.min_distance_hz < 0:
            raise ValueError("Analysis band minimum distance must be non-negative")
        if band.min_stability < 0:
            raise ValueError("Analysis band minimum stability must be non-negative")
        if band.max_frequency <= band.min_frequency:
            raise ValueError(
                "Analysis band maximum frequency must be greater than minimum frequency"
            )

    if len(median) == 0:
        return AxisPeaks(
            frequencies=np.array([]),
            amplitudes=np.array([]),
            properties={}
        )

    if len(freq) < 2:
        raise ValueError("Frequency axis must contain at least two values")

    if len(stability) != len(median):
        raise ValueError("Stability length does not match median PSD length")

    safe_median = np.maximum(
        median,
        np.finfo(float).tiny,
    )
    median_db = 10.0 * np.log10(safe_median)

    resolution = freq[1] - freq[0]
    peaks_by_index = {}
    property_dtypes = {}

    for band in analysis_bands:
        band_mask = (
            (freq >= band.min_frequency)
            & (freq <= band.max_frequency)
        )
        global_indices = np.flatnonzero(band_mask)
        if len(global_indices) < 2:
            continue

        band_median_db = median_db[band_mask]
        band_stability = stability[band_mask]
        distance = max(int(band.min_distance_hz / resolution), 1)
        local_peak_indices, properties = find_peaks(
            band_median_db,
            prominence=band.prominence_db,
            distance=distance,
        )
        property_dtypes.update(
            (name, values.dtype) for name, values in properties.items()
        )

        stable_peak_mask = (
            band_stability[local_peak_indices] >= band.min_stability
        )
        local_peak_indices = local_peak_indices[stable_peak_mask]
        properties = {
            name: values[stable_peak_mask]
            for name, values in properties.items()
        }
        global_peak_indices = global_indices[local_peak_indices]

        for position, global_peak_index in enumerate(global_peak_indices):
            peak_properties = {
                name: values[position]
                for name, values in properties.items()
            }
            for name in ("left_bases", "right_bases"):
                if name in peak_properties:
                    peak_properties[name] = global_indices[peak_properties[name]]

            candidate_prominence_db = peak_properties["prominences"]
            existing_properties = peaks_by_index.get(global_peak_index)
            existing_prominence_db = (
                existing_properties["prominences"]
                if existing_properties is not None
                else None
            )
            if (
                existing_prominence_db is None
                or candidate_prominence_db > existing_prominence_db
            ):
                peaks_by_index[global_peak_index] = peak_properties

    peak_indices = np.array(sorted(peaks_by_index), dtype=int)
    merged_properties = {
        name: np.asarray([
            peaks_by_index[peak_index][name]
            for peak_index in peak_indices
        ], dtype=dtype)
        for name, dtype in property_dtypes.items()
    }

    return AxisPeaks(
        frequencies=freq[peak_indices],
        amplitudes=median[peak_indices],
        properties=merged_properties,
    )


def find_psd_peaks(
    statistics: StatisticsResult,
    sessions: list[SessionResult],
    analysis_bands: list[AnalysisBand],
) -> PeakResult:
    if not sessions:
        raise ValueError("At least one completed session is required")
    if not analysis_bands:
        raise ValueError("At least one analysis band is required")

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

    return PeakResult(
        x=_find_axis_peaks_by_bands(
            freq, statistics.x.median, statistics.x.stability, analysis_bands
        ),
        y=_find_axis_peaks_by_bands(
            freq, statistics.y.median, statistics.y.stability, analysis_bands
        ),
        z=_find_axis_peaks_by_bands(
            freq, statistics.z.median, statistics.z.stability, analysis_bands
        ),
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

try:
    config = load_config(CONFIG_PATH)
except FileNotFoundError:
    raise SystemExit(f"Configuration file not found: {CONFIG_PATH}")
except tomllib.TOMLDecodeError as error:
    raise SystemExit(f"Invalid TOML configuration: {error}")
except (KeyError, TypeError, ValueError) as error:
    raise SystemExit(f"Invalid configuration: {error}")

analysis_bands = config.analysis_bands
analysis_min_frequency, analysis_max_frequency = (
    get_analysis_frequency_limits(analysis_bands)
)

ser = serial.Serial(
    config.serial.port,
    config.serial.baud,
    timeout=config.serial.timeout_seconds,
)

print("Connected:", config.serial.port)


# ==========================================================
# РџРѕРёСЃРє РЅР°С‡Р°Р»Р° РїР°РєРµС‚Р°
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

    nperseg = min(config.welch.nperseg, len(signal))
    noverlap = min(config.welch.noverlap, nperseg // 2)
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
        nperseg=min(config.welch.nperseg, len(signal)),
        noverlap=min(config.welch.noverlap, len(signal)//2),
        scaling="density"
    )

    mask = (
        (freq >= analysis_min_frequency)
        & (freq <= analysis_max_frequency)
    )

    freq = freq[mask]
    psd = psd[mask]

    resolution = fs / min(config.welch.nperseg, len(signal))

    return PSDResult(
        freq=freq,
        psd=psd,
        resolution=resolution
    )

def process_session(session, session_fs, number):

    if any(
        len(session[axis]) != config.session.packets_per_session
        for axis in ("X", "Y", "Z")
    ):
        raise ValueError("Cannot process incomplete session")

    if len(session_fs) != config.session.packets_per_session:
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

    if session_packets == config.session.packets_per_session:
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

    if (
        stop_requested
        and session_packets == config.session.packets_per_session
    ):
        break

print()

statistics: StatisticsResult | None = None
peaks: PeakResult | None = None
visualization_data: VisualizationData | None = None

if sessions:
    statistics = compute_statistics(sessions)
    peaks = find_psd_peaks(
        statistics,
        sessions,
        analysis_bands,
    )
    visualization_data = build_visualization_data(
        statistics,
        peaks,
        sessions,
    )

if not sessions:
    print("No completed sessions available for analysis.")
    ser.close()
    raise SystemExit(0)

if len(sessions) < config.session.min_recommended_sessions:
    print(
        f"Warning: only {len(sessions)} completed session(s); "
        f"at least {config.session.min_recommended_sessions} are recommended."
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
    constrained_layout=True,
)

visualization_axes = {
    "X": visualization_data.x,
    "Y": visualization_data.y,
    "Z": visualization_data.z,
}

for axis_index, (axis_name, axis_data) in enumerate(visualization_axes.items()):
    psd_axis = stat_axes[axis_index * 2]
    stability_axis = stat_axes[axis_index * 2 + 1]

    draw_analysis_bands(psd_axis, analysis_bands)
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
    annotate_peak_frequencies(
        psd_axis,
        axis_data.peak_frequencies,
        axis_data.peak_amplitudes,
    )
    psd_axis.set_title(f"{axis_name} axis вЂ” Median PSD")
    psd_axis.set_ylabel("PSD [gВІ/Hz]")
    psd_axis.set_xlim(analysis_min_frequency, analysis_max_frequency)
    psd_axis.grid(True, alpha=0.25, linewidth=0.6)
    if axis_index == 0:
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
    for band_index, band in enumerate(analysis_bands):
        stability_axis.hlines(
            band.min_stability,
            band.min_frequency,
            band.max_frequency,
            linestyle="--",
            linewidth=1.0,
            alpha=0.6,
            label="Minimum stability" if band_index == 0 else None,
        )
    stability_axis.scatter(
        axis_data.peak_frequencies,
        peak_stability,
        color=COLORS[axis_name],
        marker="x",
        label="Stable peaks",
    )
    stability_axis.set_title(f"{axis_name} axis вЂ” Stability")
    stability_axis.set_ylabel("Mean / Std")
    stability_axis.set_xlim(
        analysis_min_frequency,
        analysis_max_frequency,
    )
    stability_axis.grid(True, alpha=0.25, linewidth=0.6)
    if axis_index == 0:
        stability_axis.legend()

stat_axes[-1].set_xlabel("Frequency, Hz")
stat_fig.suptitle("Statistical vibration analysis")


plt.show()

