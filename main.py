import argparse
import struct
import sys
import tomllib
from contextlib import redirect_stdout
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import serial
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.signal import find_peaks

# ==========================================================
# Настройки
# ==========================================================

CONFIG_PATH = Path(__file__).with_name("config.toml")
RESULTS_DIRECTORY = Path("results")

# AXIS = "X"          # X / Y / Z

MAGIC = b"VIB2"

SUPPORTED_ODR_HZ = {250.0, 125.0, 62.5}

SET_ODR_COMMAND = 0x01
ODR_PARAMETER_BY_HZ = {
    250.0: 0x00,
    125.0: 0x01,
    62.5: 0x02,
}

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
    frequency_tolerance_hz: float
    frequency_stability_max_std_hz: float
    noise_window_hz: float


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
class SensorConfig:
    odr_hz: float


@dataclass(frozen=True)
class WelchConfig:
    nperseg: int
    noverlap: int


@dataclass(frozen=True)
class TrustedFrequencyVisualizationConfig:
    min_support_fraction: float
    min_median_prominence_db: float
    background_weight: float
    min_band_contrast_db: float
    weak_trusted_weight: float


@dataclass(frozen=True)
class FrequencyClusterConsolidationConfig:
    median_frequency_tolerance_hz: float


@dataclass(frozen=True)
class FrequencyClusteringConfig:
    frequency_tolerance_hz_250: float
    frequency_tolerance_hz_125: float
    frequency_tolerance_hz_62p5: float


@dataclass(frozen=True)
class VisualizationConfig:
    trusted_frequency: TrustedFrequencyVisualizationConfig


@dataclass(frozen=True)
class ApplicationConfig:
    serial: SerialConfig
    session: SessionConfig
    sensor: SensorConfig
    welch: WelchConfig
    visualization: VisualizationConfig
    frequency_clustering: FrequencyClusteringConfig
    frequency_cluster_consolidation: FrequencyClusterConsolidationConfig
    analysis_bands: list[AnalysisBand]


@dataclass(frozen=True)
class RunResultPaths:
    log: Path
    figure1: Path
    figure2: Path
    raw: Path


@dataclass(frozen=True)
class RawMeasurement:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    packet_fs_hz: np.ndarray
    created_at: str
    requested_odr_hz: float
    packets_per_session: int
    target_sessions: int


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

    require_finite_number(config.sensor.odr_hz, "Sensor odr_hz")
    if config.sensor.odr_hz not in SUPPORTED_ODR_HZ:
        raise ValueError(
            "Sensor odr_hz must be one of 250, 125, or 62.5 Hz"
        )

    require_positive_integer(config.welch.nperseg, "Welch nperseg")
    require_non_negative_integer(config.welch.noverlap, "Welch noverlap")
    if config.welch.noverlap >= config.welch.nperseg:
        raise ValueError("Welch noverlap must be less than nperseg")

    trusted_frequency = config.visualization.trusted_frequency
    require_finite_number(
        trusted_frequency.min_support_fraction,
        "Visualization trusted frequency min_support_fraction",
    )
    require_finite_number(
        trusted_frequency.min_median_prominence_db,
        "Visualization trusted frequency min_median_prominence_db",
    )
    require_finite_number(
        trusted_frequency.background_weight,
        "Visualization trusted frequency background_weight",
    )
    require_finite_number(
        trusted_frequency.min_band_contrast_db,
        "Visualization trusted frequency min_band_contrast_db",
    )
    require_finite_number(
        trusted_frequency.weak_trusted_weight,
        "Visualization trusted frequency weak_trusted_weight",
    )
    if not 0 < trusted_frequency.min_support_fraction <= 1:
        raise ValueError(
            "Visualization trusted frequency min_support_fraction must be "
            "greater than zero and at most one"
        )
    if trusted_frequency.min_median_prominence_db < 0:
        raise ValueError(
            "Visualization trusted frequency min_median_prominence_db must "
            "be non-negative"
        )
    if not 0 <= trusted_frequency.background_weight <= 1:
        raise ValueError(
            "Visualization trusted frequency background_weight must be "
            "between zero and one"
        )
    if trusted_frequency.min_band_contrast_db < 0:
        raise ValueError(
            "Visualization trusted frequency min_band_contrast_db must be "
            "non-negative"
        )
    if not 0 <= trusted_frequency.weak_trusted_weight <= 1:
        raise ValueError(
            "Visualization trusted frequency weak_trusted_weight must be "
            "between zero and one"
        )
    if (
        trusted_frequency.weak_trusted_weight
        < trusted_frequency.background_weight
    ):
        raise ValueError(
            "Visualization trusted frequency weak_trusted_weight must not be "
            "less than background_weight"
        )

    consolidation = config.frequency_cluster_consolidation
    require_finite_number(
        consolidation.median_frequency_tolerance_hz,
        "Frequency cluster consolidation median_frequency_tolerance_hz",
    )
    if consolidation.median_frequency_tolerance_hz <= 0:
        raise ValueError(
            "Frequency cluster consolidation "
            "median_frequency_tolerance_hz must be positive"
        )

    frequency_clustering = config.frequency_clustering
    clustering_tolerances = {
        "frequency_tolerance_hz_250": (
            frequency_clustering.frequency_tolerance_hz_250
        ),
        "frequency_tolerance_hz_125": (
            frequency_clustering.frequency_tolerance_hz_125
        ),
        "frequency_tolerance_hz_62p5": (
            frequency_clustering.frequency_tolerance_hz_62p5
        ),
    }
    for tolerance_name, tolerance in clustering_tolerances.items():
        require_finite_number(
            tolerance,
            f"Frequency clustering {tolerance_name}",
        )
        if tolerance <= 0:
            raise ValueError(
                f"Frequency clustering {tolerance_name} must be positive"
            )

    effective_frequency_tolerance_hz = resolve_frequency_tolerance_hz(
        frequency_clustering,
        config.sensor.odr_hz,
    )

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
        require_finite_number(
            band.frequency_tolerance_hz,
            f"{band_name} frequency_tolerance_hz",
        )
        require_finite_number(
            band.frequency_stability_max_std_hz,
            f"{band_name} frequency_stability_max_std_hz",
        )
        require_finite_number(
            band.noise_window_hz,
            f"{band_name} noise_window_hz",
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
        if band.frequency_tolerance_hz < 0:
            raise ValueError(
                f"{band_name} frequency_tolerance_hz must be non-negative"
            )
        if band.frequency_tolerance_hz != effective_frequency_tolerance_hz:
            raise ValueError(
                f"{band_name} frequency_tolerance_hz does not match "
                "the effective ODR-dependent tolerance"
            )
        if band.frequency_stability_max_std_hz <= 0:
            raise ValueError(
                f"{band_name} frequency_stability_max_std_hz must be positive"
            )
        if band.noise_window_hz <= 0:
            raise ValueError(
                f"{band_name} noise_window_hz must be positive"
            )
        if band.noise_window_hz <= band.frequency_tolerance_hz:
            raise ValueError(
                f"{band_name} noise_window_hz must be greater than "
                "frequency_tolerance_hz"
            )


def load_config(
    path: Path,
    *,
    validate: bool = True,
) -> ApplicationConfig:
    with path.open("rb") as file:
        raw_config = tomllib.load(file)

    serial_data = raw_config["serial"]
    session_data = raw_config["session"]
    sensor_data = raw_config["sensor"]
    welch_data = raw_config["welch"]
    trusted_frequency_data = raw_config["visualization"][
        "trusted_frequency"
    ]
    consolidation_data = raw_config["analysis"][
        "frequency_cluster_consolidation"
    ]
    frequency_clustering_data = raw_config["analysis"][
        "frequency_clustering"
    ]
    band_entries = raw_config["analysis"]["bands"]

    sensor_odr_hz = float(sensor_data["odr_hz"])
    frequency_clustering = FrequencyClusteringConfig(
        frequency_tolerance_hz_250=frequency_clustering_data[
            "frequency_tolerance_hz_250"
        ],
        frequency_tolerance_hz_125=frequency_clustering_data[
            "frequency_tolerance_hz_125"
        ],
        frequency_tolerance_hz_62p5=frequency_clustering_data[
            "frequency_tolerance_hz_62p5"
        ],
    )
    frequency_tolerance_hz = resolve_frequency_tolerance_hz(
        frequency_clustering,
        sensor_odr_hz,
    )

    analysis_bands = [
        AnalysisBand(
            name=entry["name"],
            min_frequency=entry["min_frequency"],
            max_frequency=entry["max_frequency"],
            prominence_db=entry["prominence_db"],
            min_distance_hz=entry["min_distance_hz"],
            min_stability=entry["min_stability"],
            frequency_tolerance_hz=frequency_tolerance_hz,
            frequency_stability_max_std_hz=entry[
                "frequency_stability_max_std_hz"
            ],
            noise_window_hz=entry["noise_window_hz"],
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
        sensor=SensorConfig(
            odr_hz=sensor_odr_hz,
        ),
        welch=WelchConfig(
            nperseg=welch_data["nperseg"],
            noverlap=welch_data["noverlap"],
        ),
        visualization=VisualizationConfig(
            trusted_frequency=TrustedFrequencyVisualizationConfig(
                min_support_fraction=trusted_frequency_data[
                    "min_support_fraction"
                ],
                min_median_prominence_db=trusted_frequency_data[
                    "min_median_prominence_db"
                ],
                background_weight=trusted_frequency_data[
                    "background_weight"
                ],
                min_band_contrast_db=trusted_frequency_data[
                    "min_band_contrast_db"
                ],
                weak_trusted_weight=trusted_frequency_data[
                    "weak_trusted_weight"
                ],
            ),
        ),
        frequency_clustering=frequency_clustering,
        frequency_cluster_consolidation=FrequencyClusterConsolidationConfig(
            median_frequency_tolerance_hz=consolidation_data[
                "median_frequency_tolerance_hz"
            ],
        ),
        analysis_bands=analysis_bands,
    )
    if validate:
        validate_config(config)
    return config


def parse_cli_arguments(
    arguments: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--odr", type=float)
    parser.add_argument("--packets-per-session", type=int)
    parser.add_argument("--min-recommended-sessions", type=int)
    parser.add_argument("--repeat", type=positive_integer, default=1)
    return parser.parse_args(arguments)


def positive_integer(value: str) -> int:
    try:
        parsed_value = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed_value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed_value


def normalize_cli_odr_hz(odr_hz: float) -> float:
    if odr_hz == 62.0:
        return 62.5
    return odr_hz


def resolve_frequency_tolerance_hz(
    config: FrequencyClusteringConfig,
    odr_hz: float,
) -> float:
    tolerance_by_odr = {
        250.0: config.frequency_tolerance_hz_250,
        125.0: config.frequency_tolerance_hz_125,
        62.5: config.frequency_tolerance_hz_62p5,
    }
    try:
        return float(tolerance_by_odr[odr_hz])
    except KeyError as error:
        raise ValueError(
            f"Unsupported ODR for frequency clustering: {odr_hz} Hz"
        ) from error


def apply_cli_overrides(
    config: ApplicationConfig,
    arguments: argparse.Namespace,
) -> ApplicationConfig:
    odr_hz = config.sensor.odr_hz
    if arguments.odr is not None:
        odr_hz = normalize_cli_odr_hz(arguments.odr)

    frequency_tolerance_hz = resolve_frequency_tolerance_hz(
        config.frequency_clustering,
        odr_hz,
    )
    effective_config = replace(
        config,
        sensor=replace(config.sensor, odr_hz=odr_hz),
        analysis_bands=[
            replace(
                band,
                frequency_tolerance_hz=frequency_tolerance_hz,
            )
            for band in config.analysis_bands
        ],
        session=replace(
            config.session,
            packets_per_session=(
                config.session.packets_per_session
                if arguments.packets_per_session is None
                else arguments.packets_per_session
            ),
            min_recommended_sessions=(
                config.session.min_recommended_sessions
                if arguments.min_recommended_sessions is None
                else arguments.min_recommended_sessions
            ),
        ),
    )
    validate_config(effective_config)
    return effective_config


def build_effective_config(
    path: Path,
    arguments: list[str] | None = None,
) -> ApplicationConfig:
    config = load_config(path, validate=False)
    cli_arguments = parse_cli_arguments(arguments)
    return apply_cli_overrides(config, cli_arguments)


def build_effective_config_from_cli(
    path: Path,
    cli_arguments: argparse.Namespace,
) -> ApplicationConfig:
    config = load_config(path, validate=False)
    return apply_cli_overrides(config, cli_arguments)


def build_set_odr_command(odr_hz: float) -> bytes:
    try:
        parameter = ODR_PARAMETER_BY_HZ[odr_hz]
    except KeyError as error:
        raise ValueError(f"Unsupported ADXL355 ODR: {odr_hz} Hz") from error
    return bytes([SET_ODR_COMMAND, parameter])


def send_adxl355_odr_command(
    serial_port,
    config: ApplicationConfig,
) -> None:
    serial_port.write(build_set_odr_command(config.sensor.odr_hz))
    serial_port.flush()


def format_odr_for_filename(odr_hz: float) -> str:
    return f"{odr_hz:g}".replace(".", "p")


def build_run_result_paths(
    run_started_at: datetime,
    odr_hz: float,
    run_number: int,
) -> RunResultPaths:
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    timestamp = run_started_at.strftime("%Y%m%d_%H%M%S")
    base_name = (
        f"{timestamp}_ODR{format_odr_for_filename(odr_hz)}_"
        f"run{run_number:02d}"
    )

    collision_number = 1
    while True:
        suffix = "" if collision_number == 1 else f"_collision{collision_number:02d}"
        candidate_base = f"{base_name}{suffix}"
        paths = RunResultPaths(
            log=RESULTS_DIRECTORY / f"{candidate_base}.txt",
            figure1=RESULTS_DIRECTORY / f"{candidate_base}_figure1.png",
            figure2=RESULTS_DIRECTORY / f"{candidate_base}_figure2.png",
            raw=RESULTS_DIRECTORY / f"{candidate_base}_raw.npz",
        )
        if not any(
            path.exists()
            for path in (paths.log, paths.figure1, paths.figure2, paths.raw)
        ):
            return paths
        collision_number += 1


def initialize_run_log(
    paths: RunResultPaths,
    run_started_at: datetime,
    config: ApplicationConfig,
    run_number: int,
    total_runs: int,
    packet_count: int,
    duration_seconds: float,
    measured_fs: float,
    raw_packet_count: int | None = None,
    raw_samples_per_packet: int | None = None,
) -> None:
    with paths.log.open("x", encoding="utf-8", newline="\n") as run_log:
        run_log.write(
            f"Run started: {run_started_at:%Y-%m-%d %H:%M:%S}\n"
            f"Run: {run_number}/{total_runs}\n"
            f"ODR: {config.sensor.odr_hz:g} Hz\n"
            f"Packets/session: {config.session.packets_per_session}\n"
            f"Target sessions: {config.session.min_recommended_sessions}\n"
            "Frequency tolerance: "
            f"{resolve_frequency_tolerance_hz(config.frequency_clustering, config.sensor.odr_hz):.2f} "
            "Hz\n"
            f"Welch nperseg: {config.welch.nperseg}\n"
            f"Welch noverlap: {config.welch.noverlap}\n"
            "\n"
            f"Packets: {packet_count:4d}"
            f"   Duration: {duration_seconds:6.1f} s"
            f"   Fs={measured_fs:6.2f}\n"
        )
        if raw_packet_count is not None:
            if raw_samples_per_packet is None:
                raise ValueError(
                    "Raw samples per packet are required with raw packet count"
                )
            run_log.write(
                f"Raw data: {paths.raw.name}\n"
                f"Raw packets: {raw_packet_count}\n"
                f"Samples/packet: {raw_samples_per_packet}\n"
            )


def validate_raw_measurement(raw: RawMeasurement) -> None:
    packet_arrays = {
        "x": raw.x,
        "y": raw.y,
        "z": raw.z,
    }
    for axis_name, values in packet_arrays.items():
        if not isinstance(values, np.ndarray) or values.ndim != 2:
            raise ValueError(
                f"Raw {axis_name} data must be a two-dimensional NumPy array"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Raw {axis_name} data must be finite")
    if raw.x.shape != raw.y.shape or raw.x.shape != raw.z.shape:
        raise ValueError("Raw X/Y/Z data shapes must match")
    packet_count, samples_per_packet = raw.x.shape
    if packet_count == 0 or samples_per_packet == 0:
        raise ValueError("Raw measurement must contain packets and samples")
    if (
        not isinstance(raw.packet_fs_hz, np.ndarray)
        or raw.packet_fs_hz.shape != (packet_count,)
    ):
        raise ValueError("Raw packet_fs_hz shape must match packet count")
    if (
        not np.all(np.isfinite(raw.packet_fs_hz))
        or np.any(raw.packet_fs_hz <= 0)
    ):
        raise ValueError("Raw packet_fs_hz must be positive and finite")
    if not isinstance(raw.created_at, str) or not raw.created_at:
        raise ValueError("Raw created_at must be a non-empty string")
    if raw.requested_odr_hz not in SUPPORTED_ODR_HZ:
        raise ValueError("Raw requested ODR is unsupported")
    if raw.packets_per_session <= 0 or raw.target_sessions <= 0:
        raise ValueError("Raw session dimensions must be positive")
    expected_packet_count = raw.packets_per_session * raw.target_sessions
    if packet_count != expected_packet_count:
        raise ValueError(
            "Raw packet count must match packets_per_session * target_sessions"
        )


def save_raw_measurement(path: Path, raw: RawMeasurement) -> None:
    validate_raw_measurement(raw)
    with path.open("xb") as raw_file:
        np.savez_compressed(
            raw_file,
            format_version=np.asarray(1, dtype=np.int64),
            created_at=np.asarray(raw.created_at),
            requested_odr_hz=np.asarray(raw.requested_odr_hz, dtype=float),
            packet_count=np.asarray(raw.x.shape[0], dtype=np.int64),
            samples_per_packet=np.asarray(raw.x.shape[1], dtype=np.int64),
            packets_per_session=np.asarray(
                raw.packets_per_session,
                dtype=np.int64,
            ),
            target_sessions=np.asarray(raw.target_sessions, dtype=np.int64),
            x=raw.x,
            y=raw.y,
            z=raw.z,
            packet_fs_hz=raw.packet_fs_hz,
        )


def load_raw_measurement(path: Path) -> RawMeasurement:
    with np.load(path, allow_pickle=False) as archive:
        if int(archive["format_version"]) != 1:
            raise ValueError("Unsupported raw measurement format version")
        raw = RawMeasurement(
            x=np.asarray(archive["x"]),
            y=np.asarray(archive["y"]),
            z=np.asarray(archive["z"]),
            packet_fs_hz=np.asarray(archive["packet_fs_hz"]),
            created_at=str(archive["created_at"]),
            requested_odr_hz=float(archive["requested_odr_hz"]),
            packets_per_session=int(archive["packets_per_session"]),
            target_sessions=int(archive["target_sessions"]),
        )
        if int(archive["packet_count"]) != raw.x.shape[0]:
            raise ValueError("Raw packet_count metadata does not match data")
        if int(archive["samples_per_packet"]) != raw.x.shape[1]:
            raise ValueError("Raw samples_per_packet metadata does not match data")
    validate_raw_measurement(raw)
    return raw


class TeeTextOutput:
    def __init__(self, *outputs) -> None:
        self.outputs = outputs

    def write(self, text: str) -> int:
        for output in self.outputs:
            output.write(text)
        return len(text)

    def flush(self) -> None:
        for output in self.outputs:
            output.flush()


def append_run_diagnostics(
    log_path: Path,
    printer,
    *args,
) -> None:
    with log_path.open("a", encoding="utf-8", newline="\n") as run_log:
        with redirect_stdout(TeeTextOutput(sys.stdout, run_log)):
            printer(*args)


def save_run_figures(
    paths: RunResultPaths,
    stat_fig,
    trusted_fig,
) -> None:
    with paths.figure1.open("xb") as figure1_file:
        stat_fig.savefig(figure1_file, format="png", dpi=100)
    with paths.figure2.open("xb") as figure2_file:
        trusted_fig.savefig(figure2_file, format="png", dpi=100)


def get_analysis_frequency_limits(
    analysis_bands: list[AnalysisBand],
) -> tuple[float, float]:
    return (
        min(band.min_frequency for band in analysis_bands),
        max(band.max_frequency for band in analysis_bands),
    )


def build_frequency_window_mask(
    frequency: np.ndarray,
    center_frequency: float,
    tolerance_hz: float,
) -> np.ndarray:
    if not isinstance(frequency, np.ndarray) or frequency.ndim != 1:
        raise ValueError("Frequency must be a one-dimensional array")
    if frequency.size == 0:
        raise ValueError("Frequency array must not be empty")

    try:
        frequency_is_finite = np.all(np.isfinite(frequency))
    except TypeError:
        frequency_is_finite = False
    if not frequency_is_finite:
        raise ValueError("Frequency must contain only finite values")
    if np.any(np.diff(frequency) <= 0):
        raise ValueError("Frequency must be strictly increasing")

    valid_number_types = (int, float, np.integer, np.floating)
    invalid_boolean_types = (bool, np.bool_)
    if (
        not isinstance(center_frequency, valid_number_types)
        or isinstance(center_frequency, invalid_boolean_types)
        or not np.isfinite(center_frequency)
    ):
        raise ValueError("Center frequency must be a finite number")
    if (
        not isinstance(tolerance_hz, valid_number_types)
        or isinstance(tolerance_hz, invalid_boolean_types)
        or not np.isfinite(tolerance_hz)
    ):
        raise ValueError("Frequency tolerance must be a finite number")
    if tolerance_hz < 0:
        raise ValueError("Frequency tolerance must be non-negative")

    window_mask = (
        frequency >= center_frequency - tolerance_hz
    ) & (
        frequency <= center_frequency + tolerance_hz
    )
    if not np.any(window_mask):
        raise ValueError("Frequency window must contain at least one point")

    return window_mask


def build_local_noise_mask(
    frequency: np.ndarray,
    peak_frequency: float,
    peak_tolerance_hz: float,
    noise_window_hz: float,
    band_min_frequency: float,
    band_max_frequency: float,
) -> np.ndarray:
    if not isinstance(frequency, np.ndarray) or frequency.ndim != 1:
        raise ValueError("Frequency must be a one-dimensional array")
    if len(frequency) < 2:
        raise ValueError("Frequency axis must contain at least two points")

    try:
        if not np.all(np.isfinite(frequency)):
            raise ValueError("Frequency must contain only finite values")
        if not np.all(np.diff(frequency) > 0):
            raise ValueError("Frequency must be strictly increasing")
    except TypeError as error:
        raise ValueError("Frequency must contain numeric values") from error

    scalar_parameters = {
        "Peak frequency": peak_frequency,
        "Peak tolerance": peak_tolerance_hz,
        "Noise window": noise_window_hz,
        "Band minimum frequency": band_min_frequency,
        "Band maximum frequency": band_max_frequency,
    }
    valid_number_types = (int, float, np.integer, np.floating)
    invalid_boolean_types = (bool, np.bool_)
    for name, value in scalar_parameters.items():
        if (
            not isinstance(value, valid_number_types)
            or isinstance(value, invalid_boolean_types)
            or not np.isfinite(value)
        ):
            raise ValueError(f"{name} must be a finite number")

    if peak_tolerance_hz < 0:
        raise ValueError("Peak tolerance must be non-negative")
    if noise_window_hz < 0:
        raise ValueError("Noise window must be non-negative")
    if noise_window_hz <= peak_tolerance_hz:
        raise ValueError("Noise window must be greater than peak tolerance")
    if band_max_frequency <= band_min_frequency:
        raise ValueError(
            "Band maximum frequency must be greater than minimum frequency"
        )
    if not band_min_frequency <= peak_frequency <= band_max_frequency:
        raise ValueError("Peak frequency must be inside the analysis band")

    outer_min_frequency = max(
        peak_frequency - noise_window_hz,
        band_min_frequency,
    )
    outer_max_frequency = min(
        peak_frequency + noise_window_hz,
        band_max_frequency,
    )
    outer_mask = (
        (frequency >= outer_min_frequency)
        & (frequency <= outer_max_frequency)
    )
    peak_window_mask = (
        (frequency >= peak_frequency - peak_tolerance_hz)
        & (frequency <= peak_frequency + peak_tolerance_hz)
    )
    return outer_mask & ~peak_window_mask


def compute_local_snr_db(
    median_psd: np.ndarray,
    peak_index: int,
    noise_mask: np.ndarray,
) -> tuple[float, float]:
    if not isinstance(median_psd, np.ndarray) or median_psd.ndim != 1:
        raise ValueError("Median PSD must be a one-dimensional array")
    try:
        if not np.all(np.isfinite(median_psd)):
            raise ValueError("Median PSD must contain only finite values")
        if np.any(median_psd < 0):
            raise ValueError("Median PSD must not contain negative values")
    except TypeError as error:
        raise ValueError("Median PSD must contain numeric values") from error

    if not isinstance(noise_mask, np.ndarray) or noise_mask.ndim != 1:
        raise ValueError("Noise mask must be a one-dimensional array")
    if noise_mask.dtype != np.bool_:
        raise ValueError("Noise mask must have boolean dtype")
    if len(noise_mask) != len(median_psd):
        raise ValueError("Median PSD and noise mask lengths must match")

    if (
        not isinstance(peak_index, (int, np.integer))
        or isinstance(peak_index, (bool, np.bool_))
    ):
        raise ValueError("Peak index must be an integer")
    if not 0 <= peak_index < len(median_psd):
        raise ValueError("Peak index is outside Median PSD")
    if noise_mask[peak_index]:
        raise ValueError("Peak index must not be included in the noise mask")
    if np.count_nonzero(noise_mask) < 3:
        raise ValueError("Noise region must contain at least three points")

    local_noise_floor = float(np.median(median_psd[noise_mask]))
    peak_psd = median_psd[peak_index]
    safe_peak = max(peak_psd, np.finfo(float).tiny)
    safe_noise = max(local_noise_floor, np.finfo(float).tiny)
    local_snr_db = float(10.0 * np.log10(safe_peak / safe_noise))
    return local_noise_floor, local_snr_db


def compute_local_psd_background(
    frequency: np.ndarray,
    median_psd: np.ndarray,
    analysis_bands: list[AnalysisBand],
) -> np.ndarray:
    if not isinstance(frequency, np.ndarray) or frequency.ndim != 1:
        raise ValueError("Frequency must be a one-dimensional array")
    if len(frequency) < 2:
        raise ValueError("Frequency axis must contain at least two points")
    try:
        if not np.all(np.isfinite(frequency)):
            raise ValueError("Frequency must contain only finite values")
        if not np.all(np.diff(frequency) > 0):
            raise ValueError("Frequency must be strictly increasing")
    except TypeError as error:
        raise ValueError("Frequency must contain numeric values") from error

    if not isinstance(median_psd, np.ndarray) or median_psd.ndim != 1:
        raise ValueError("Median PSD must be a one-dimensional array")
    if len(median_psd) != len(frequency):
        raise ValueError("Frequency and Median PSD lengths must match")
    try:
        if not np.all(np.isfinite(median_psd)):
            raise ValueError("Median PSD must contain only finite values")
        if np.any(median_psd < 0):
            raise ValueError("Median PSD must not contain negative values")
    except TypeError as error:
        raise ValueError("Median PSD must contain numeric values") from error

    if not isinstance(analysis_bands, list) or not analysis_bands:
        raise ValueError("At least one analysis band is required")

    local_background = np.full(len(frequency), np.nan, dtype=float)
    for index, center_frequency in enumerate(frequency):
        band = next(
            (
                candidate
                for candidate in analysis_bands
                if candidate.min_frequency
                <= center_frequency
                <= candidate.max_frequency
            ),
            None,
        )
        if band is None:
            continue

        noise_mask = build_local_noise_mask(
            frequency,
            center_frequency,
            band.frequency_tolerance_hz,
            band.noise_window_hz,
            band.min_frequency,
            band.max_frequency,
        )
        if np.count_nonzero(noise_mask) < 3:
            continue
        local_background[index] = np.median(median_psd[noise_mask])

    return local_background


def compute_local_psd_contrast_db(
    median_psd: np.ndarray,
    local_background: np.ndarray,
) -> np.ndarray:
    if not isinstance(median_psd, np.ndarray) or median_psd.ndim != 1:
        raise ValueError("Median PSD must be a one-dimensional array")
    if not isinstance(local_background, np.ndarray) or local_background.ndim != 1:
        raise ValueError("Local PSD background must be a one-dimensional array")
    if len(median_psd) != len(local_background):
        raise ValueError("Median PSD and local background lengths must match")

    try:
        if not np.all(np.isfinite(median_psd)):
            raise ValueError("Median PSD must contain only finite values")
        finite_background = local_background[~np.isnan(local_background)]
        if not np.all(np.isfinite(finite_background)):
            raise ValueError(
                "Local PSD background must contain only finite values or NaN"
            )
        if np.any(median_psd < 0):
            raise ValueError("Median PSD must not contain negative values")
        if np.any(finite_background < 0):
            raise ValueError("Local PSD background must not contain negative values")
    except TypeError as error:
        raise ValueError("PSD arrays must contain numeric values") from error

    tiny = np.finfo(float).tiny
    safe_psd = np.maximum(median_psd, tiny)
    safe_background = np.maximum(local_background, tiny)
    return 10.0 * np.log10(safe_psd / safe_background)


def compute_window_power(
    frequency: np.ndarray,
    psd: np.ndarray,
    window_mask: np.ndarray,
) -> float:
    arrays = {
        "Frequency": frequency,
        "PSD": psd,
        "Frequency window mask": window_mask,
    }
    for name, array in arrays.items():
        if not isinstance(array, np.ndarray) or array.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional array")

    if len(frequency) != len(psd) or len(frequency) != len(window_mask):
        raise ValueError("Frequency, PSD, and window mask lengths must match")
    if len(frequency) < 2:
        raise ValueError("Frequency axis must contain at least two points")
    if window_mask.dtype != np.bool_:
        raise ValueError("Frequency window mask must have boolean dtype")
    if not np.all(np.isfinite(psd)):
        raise ValueError("PSD must contain only finite values")
    if np.any(psd < 0):
        raise ValueError("PSD must not contain negative values")
    window_point_count = np.count_nonzero(window_mask)
    if window_point_count == 0:
        raise ValueError("Frequency window must contain at least one point")

    if window_point_count == 1:
        index = np.flatnonzero(window_mask)[0]
        if index == 0:
            bin_width = frequency[1] - frequency[0]
        elif index == len(frequency) - 1:
            bin_width = frequency[-1] - frequency[-2]
        else:
            bin_width = (
                frequency[index + 1]
                - frequency[index - 1]
            ) / 2
        window_power = psd[index] * bin_width
    else:
        window_power = np.trapezoid(
            psd[window_mask],
            frequency[window_mask],
        )
    if not np.isfinite(window_power):
        raise ValueError("Frequency window power must be finite")
    if window_power < 0:
        raise ValueError("Frequency window power must be non-negative")

    return float(window_power)


def compute_local_window_power_stability(
    frequency: np.ndarray,
    session_psd_stack: np.ndarray,
    analysis_bands: list[AnalysisBand],
) -> np.ndarray:
    if not isinstance(frequency, np.ndarray) or frequency.ndim != 1:
        raise ValueError("Frequency must be a one-dimensional NumPy array")
    if len(frequency) == 0:
        raise ValueError("Frequency must not be empty")
    try:
        if not np.all(np.isfinite(frequency)):
            raise ValueError("Frequency must contain only finite values")
        if not np.all(np.diff(frequency) > 0):
            raise ValueError("Frequency must be strictly increasing")
    except TypeError as error:
        raise ValueError("Frequency must contain numeric values") from error

    if not isinstance(session_psd_stack, np.ndarray) or session_psd_stack.ndim != 2:
        raise ValueError("Session PSD stack must be a two-dimensional NumPy array")
    if session_psd_stack.shape[0] == 0:
        raise ValueError("Session PSD stack must contain at least one session")
    if session_psd_stack.shape[1] != len(frequency):
        raise ValueError("Session PSD stack width must match frequency length")
    try:
        if not np.all(np.isfinite(session_psd_stack)):
            raise ValueError("Session PSD stack must contain only finite values")
        if np.any(session_psd_stack < 0):
            raise ValueError("Session PSD stack must not contain negative values")
    except TypeError as error:
        raise ValueError("Session PSD stack must contain numeric values") from error

    if not isinstance(analysis_bands, list) or not analysis_bands:
        raise ValueError("At least one analysis band is required")

    local_stability = np.empty(len(frequency), dtype=float)
    for frequency_index, center_frequency in enumerate(frequency):
        band = next(
            (
                candidate_band
                for candidate_band in analysis_bands
                if candidate_band.min_frequency
                <= center_frequency
                <= candidate_band.max_frequency
            ),
            None,
        )
        if band is None:
            raise ValueError(
                f"Frequency {center_frequency} Hz is outside all analysis bands"
            )

        window_mask = build_frequency_window_mask(
            frequency,
            center_frequency,
            band.frequency_tolerance_hz,
        )
        window_powers = np.asarray([
            compute_window_power(frequency, session_psd, window_mask)
            for session_psd in session_psd_stack
        ])
        power_mean = np.mean(window_powers)
        power_std = np.std(window_powers)
        local_stability[frequency_index] = np.divide(
            power_mean,
            power_std,
            out=np.array(0.0),
            where=power_std != 0,
        )

    if not np.all(np.isfinite(local_stability)):
        raise ValueError("Local window power stability must contain only finite values")
    if np.any(local_stability < 0):
        raise ValueError("Local window power stability must be non-negative")
    return local_stability


def find_session_peak_in_window(
    frequency: np.ndarray,
    psd: np.ndarray,
    window_mask: np.ndarray,
) -> tuple[float, float]:
    arrays = {
        "Frequency": frequency,
        "PSD": psd,
        "Frequency window mask": window_mask,
    }
    for name, array in arrays.items():
        if not isinstance(array, np.ndarray) or array.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional array")

    if len(frequency) != len(psd) or len(frequency) != len(window_mask):
        raise ValueError("Frequency, PSD, and window mask lengths must match")
    if window_mask.dtype != np.bool_:
        raise ValueError("Frequency window mask must have boolean dtype")
    if not np.any(window_mask):
        raise ValueError("Frequency window must contain at least one point")
    if not np.all(np.isfinite(frequency)):
        raise ValueError("Frequency must contain only finite values")
    if not np.all(np.isfinite(psd)):
        raise ValueError("PSD must contain only finite values")
    if np.any(psd < 0):
        raise ValueError("PSD must not contain negative values")

    window_indices = np.flatnonzero(window_mask)
    local_index = np.argmax(psd[window_mask])
    global_index = window_indices[local_index]

    return float(frequency[global_index]), float(psd[global_index])


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
class AlignedPSDData:
    frequency: np.ndarray
    x_stack: np.ndarray
    y_stack: np.ndarray
    z_stack: np.ndarray


def validate_aligned_psd_data(
    aligned: AlignedPSDData,
) -> None:
    frequency = aligned.frequency
    if not isinstance(frequency, np.ndarray) or frequency.ndim != 1:
        raise ValueError("Aligned PSD frequency must be a one-dimensional array")
    if len(frequency) < 2:
        raise ValueError("Aligned PSD frequency must contain at least two points")
    try:
        frequency_is_finite = np.all(np.isfinite(frequency))
        frequency_is_increasing = np.all(np.diff(frequency) > 0)
    except TypeError:
        frequency_is_finite = False
        frequency_is_increasing = False
    if not frequency_is_finite:
        raise ValueError("Aligned PSD frequency must contain only finite values")
    if not frequency_is_increasing:
        raise ValueError("Aligned PSD frequency must be strictly increasing")

    def validate_stack(
        stack: np.ndarray,
        axis_name: str,
    ) -> None:
        if not isinstance(stack, np.ndarray) or stack.ndim != 2:
            raise ValueError(
                f"Aligned {axis_name} PSD stack must be a two-dimensional array"
            )
        if stack.shape[0] < 1:
            raise ValueError(
                f"Aligned {axis_name} PSD stack must contain at least one session"
            )
        if stack.shape[1] != len(frequency):
            raise ValueError(
                f"Aligned {axis_name} PSD stack width must match frequency length"
            )
        try:
            stack_is_finite = np.all(np.isfinite(stack))
            stack_is_non_negative = not np.any(stack < 0)
        except TypeError:
            stack_is_finite = False
            stack_is_non_negative = False
        if not stack_is_finite:
            raise ValueError(
                f"Aligned {axis_name} PSD stack must contain only finite values"
            )
        if not stack_is_non_negative:
            raise ValueError(
                f"Aligned {axis_name} PSD stack must not contain negative values"
            )

    validate_stack(aligned.x_stack, "X")
    validate_stack(aligned.y_stack, "Y")
    validate_stack(aligned.z_stack, "Z")

    if aligned.x_stack.shape != aligned.y_stack.shape:
        raise ValueError("Aligned X and Y PSD stack shapes must match")
    if aligned.x_stack.shape != aligned.z_stack.shape:
        raise ValueError("Aligned X and Z PSD stack shapes must match")


def build_aligned_psd_data(
    sessions: list[SessionResult],
) -> AlignedPSDData:
    if not sessions:
        raise ValueError("At least one completed session is required")

    source_data: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        "x": [],
        "y": [],
        "z": [],
    }
    all_frequencies = []

    for session_index, session in enumerate(sessions, start=1):
        for axis_name in source_data:
            try:
                psd_result = getattr(session, axis_name).psd
                frequency = psd_result.freq
                psd = psd_result.psd
            except AttributeError as error:
                raise ValueError(
                    f"Session {session_index} must contain "
                    f"{axis_name.upper()} PSD data"
                ) from error

            if not isinstance(frequency, np.ndarray) or frequency.ndim != 1:
                raise ValueError(
                    f"Session {session_index} {axis_name.upper()} PSD frequency "
                    "must be a one-dimensional array"
                )
            if len(frequency) < 2:
                raise ValueError(
                    f"Session {session_index} {axis_name.upper()} PSD frequency "
                    "must contain at least two points"
                )
            try:
                frequency_is_finite = np.all(np.isfinite(frequency))
                frequency_is_increasing = np.all(np.diff(frequency) > 0)
            except TypeError:
                frequency_is_finite = False
                frequency_is_increasing = False
            if not frequency_is_finite:
                raise ValueError(
                    f"Session {session_index} {axis_name.upper()} PSD frequency "
                    "must contain only finite values"
                )
            if not frequency_is_increasing:
                raise ValueError(
                    f"Session {session_index} {axis_name.upper()} PSD frequency "
                    "must be strictly increasing"
                )

            if not isinstance(psd, np.ndarray) or psd.ndim != 1:
                raise ValueError(
                    f"Session {session_index} {axis_name.upper()} PSD "
                    "must be a one-dimensional array"
                )
            if len(psd) != len(frequency):
                raise ValueError(
                    f"Session {session_index} {axis_name.upper()} PSD length "
                    "must match frequency length"
                )
            try:
                psd_is_finite = np.all(np.isfinite(psd))
                psd_is_non_negative = not np.any(psd < 0)
            except TypeError:
                psd_is_finite = False
                psd_is_non_negative = False
            if not psd_is_finite:
                raise ValueError(
                    f"Session {session_index} {axis_name.upper()} PSD "
                    "must contain only finite values"
                )
            if not psd_is_non_negative:
                raise ValueError(
                    f"Session {session_index} {axis_name.upper()} PSD "
                    "must not contain negative values"
                )

            source_data[axis_name].append((frequency, psd))
            all_frequencies.append(frequency)

    common_min = max(frequency[0] for frequency in all_frequencies)
    common_max = min(frequency[-1] for frequency in all_frequencies)
    if common_max <= common_min:
        raise ValueError("Session PSD frequency ranges must overlap")

    first_frequency = sessions[0].x.psd.freq
    reference_mask = (
        (first_frequency >= common_min)
        & (first_frequency <= common_max)
    )
    reference_frequency = first_frequency[reference_mask]
    if len(reference_frequency) < 2:
        raise ValueError(
            "Common session PSD frequency grid must contain at least two points"
        )

    aligned_stacks = {}
    for axis_name, axis_source_data in source_data.items():
        aligned_psd_values = []
        for source_frequency, source_psd in axis_source_data:
            if (
                reference_frequency[0] < source_frequency[0]
                or reference_frequency[-1] > source_frequency[-1]
            ):
                raise ValueError(
                    "Common PSD frequency grid must be inside every source range"
                )
            aligned_psd_values.append(
                np.interp(
                    reference_frequency,
                    source_frequency,
                    source_psd,
                )
            )
        aligned_stacks[axis_name] = np.stack(
            aligned_psd_values,
            axis=0,
        )

    aligned = AlignedPSDData(
        frequency=reference_frequency,
        x_stack=aligned_stacks["x"],
        y_stack=aligned_stacks["y"],
        z_stack=aligned_stacks["z"],
    )
    validate_aligned_psd_data(aligned)
    return aligned


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
class PeakDiagnostics:
    window_power_stability: np.ndarray
    mean_session_frequencies: np.ndarray
    frequency_std_hz: np.ndarray
    minimum_session_frequencies: np.ndarray
    maximum_session_frequencies: np.ndarray
    local_noise_floor: np.ndarray
    local_snr_db: np.ndarray


@dataclass
class AxisPeaks:
    frequencies: np.ndarray
    amplitudes: np.ndarray
    diagnostics: PeakDiagnostics
    properties: dict[str, Any]


@dataclass
class PeakResult:
    x: AxisPeaks
    y: AxisPeaks
    z: AxisPeaks


@dataclass(frozen=True)
class PeakCandidateDiagnostic:
    band_name: str
    min_stability: float
    frequency_stability_max_std_hz: float
    frequency: float
    prominence_db: float
    window_power_stability: float
    local_noise_floor: float
    local_snr_db: float
    mean_session_frequency: float
    frequency_std_hz: float
    frequency_stability_passed: bool | None
    minimum_session_frequency: float
    maximum_session_frequency: float
    accepted: bool
    rejection_reason: str | None


@dataclass(frozen=True)
class PeakCandidateDiagnostics:
    x: list[PeakCandidateDiagnostic]
    y: list[PeakCandidateDiagnostic]
    z: list[PeakCandidateDiagnostic]


@dataclass(frozen=True)
class SessionPeak:
    session_index: int
    band_name: str
    frequency: float
    prominence_db: float


@dataclass(frozen=True)
class FrequencyCluster:
    band_name: str
    frequency: float
    support_count: int
    support_fraction: float
    frequency_std_hz: float
    minimum_frequency: float
    maximum_frequency: float
    session_indices: tuple[int, ...]


@dataclass(frozen=True)
class FrequencyClusterResult:
    x: list[FrequencyCluster]
    y: list[FrequencyCluster]
    z: list[FrequencyCluster]


@dataclass(frozen=True)
class MedianPSDEvidence:
    peak_frequency: float | None
    peak_psd: float | None
    prominence_db: float | None
    local_contrast_db: float | None
    band_contrast_db: float | None
    passed_prominence: bool | None


@dataclass(frozen=True)
class FrequencyClusterDiagnostic:
    cluster: FrequencyCluster
    median_evidence: MedianPSDEvidence


@dataclass(frozen=True)
class FrequencyClusterDiagnostics:
    x: list[FrequencyClusterDiagnostic]
    y: list[FrequencyClusterDiagnostic]
    z: list[FrequencyClusterDiagnostic]


@dataclass(frozen=True)
class ConsolidatedFrequencyRegion:
    band_name: str
    frequency: float
    support_count: int
    support_fraction: float
    minimum_frequency: float
    maximum_frequency: float
    session_indices: tuple[int, ...]
    median_evidence: MedianPSDEvidence
    source_clusters: tuple[FrequencyClusterDiagnostic, ...]


@dataclass(frozen=True)
class ConsolidatedFrequencyRegions:
    x: list[ConsolidatedFrequencyRegion]
    y: list[ConsolidatedFrequencyRegion]
    z: list[ConsolidatedFrequencyRegion]


def validate_frequency_cluster_consolidation_config(
    config: FrequencyClusterConsolidationConfig,
) -> None:
    if not isinstance(config, FrequencyClusterConsolidationConfig):
        raise ValueError(
            "Frequency cluster consolidation config must be a "
            "FrequencyClusterConsolidationConfig"
        )
    tolerance = config.median_frequency_tolerance_hz
    if (
        not isinstance(tolerance, (int, float))
        or isinstance(tolerance, bool)
        or not np.isfinite(tolerance)
    ):
        raise ValueError(
            "Frequency cluster consolidation tolerance must be finite"
        )
    if tolerance <= 0:
        raise ValueError(
            "Frequency cluster consolidation tolerance must be positive"
        )


def consolidate_frequency_cluster_diagnostics(
    diagnostics: list[FrequencyClusterDiagnostic],
    analysis_bands: list[AnalysisBand],
    total_sessions: int,
    config: FrequencyClusterConsolidationConfig,
) -> list[ConsolidatedFrequencyRegion]:
    if not isinstance(diagnostics, list):
        raise ValueError("Frequency cluster diagnostics must be a list")
    if not isinstance(analysis_bands, list) or not analysis_bands:
        raise ValueError("At least one analysis band is required")
    if not all(isinstance(band, AnalysisBand) for band in analysis_bands):
        raise ValueError("Analysis bands must contain AnalysisBand values")
    if (
        not isinstance(total_sessions, int)
        or isinstance(total_sessions, bool)
        or total_sessions <= 0
    ):
        raise ValueError("Total sessions must be a positive integer")
    validate_frequency_cluster_consolidation_config(config)

    band_order = {
        band.name: band_index
        for band_index, band in enumerate(analysis_bands)
    }
    for diagnostic in diagnostics:
        if not isinstance(diagnostic, FrequencyClusterDiagnostic):
            raise ValueError(
                "Frequency cluster diagnostics must contain "
                "FrequencyClusterDiagnostic values"
            )
        if diagnostic.cluster.band_name not in band_order:
            raise ValueError(
                f"Frequency cluster band "
                f"{diagnostic.cluster.band_name!r} is not configured"
            )

    sorted_diagnostics = sorted(
        diagnostics,
        key=lambda diagnostic: (
            band_order[diagnostic.cluster.band_name],
            (
                diagnostic.median_evidence.peak_frequency
                if diagnostic.median_evidence.peak_frequency is not None
                else float("inf")
            ),
            diagnostic.cluster.frequency,
        ),
    )
    source_groups: list[list[FrequencyClusterDiagnostic]] = []
    tolerance = config.median_frequency_tolerance_hz
    for diagnostic in sorted_diagnostics:
        diagnostic_frequency = diagnostic.median_evidence.peak_frequency
        diagnostic_sessions = set(diagnostic.cluster.session_indices)
        merged = False
        for source_group in source_groups:
            if (
                source_group[0].cluster.band_name
                != diagnostic.cluster.band_name
            ):
                continue
            group_frequencies = [
                source.median_evidence.peak_frequency
                for source in source_group
            ]
            if (
                diagnostic_frequency is None
                or any(frequency is None for frequency in group_frequencies)
            ):
                continue
            group_sessions = {
                session_index
                for source in source_group
                for session_index in source.cluster.session_indices
            }
            if not diagnostic_sessions.isdisjoint(group_sessions):
                continue
            frequencies = [
                float(frequency) for frequency in group_frequencies
            ] + [diagnostic_frequency]
            if max(frequencies) - min(frequencies) > tolerance:
                continue
            source_group.append(diagnostic)
            merged = True
            break
        if not merged:
            source_groups.append([diagnostic])

    consolidated_regions = []
    for source_group in source_groups:
        source_clusters = tuple(source_group)
        session_indices = tuple(sorted({
            session_index
            for source in source_clusters
            for session_index in source.cluster.session_indices
        }))
        median_frequencies = [
            source.median_evidence.peak_frequency
            for source in source_clusters
            if source.median_evidence.peak_frequency is not None
        ]
        representative_frequency = (
            float(np.median(median_frequencies))
            if median_frequencies
            else None
        )

        def evidence_key(
            source: FrequencyClusterDiagnostic,
        ) -> tuple[float, float, float, float]:
            evidence = source.median_evidence
            prominence = evidence.prominence_db
            peak_frequency = evidence.peak_frequency
            return (
                -prominence if prominence is not None else float("inf"),
                (
                    abs(peak_frequency - representative_frequency)
                    if peak_frequency is not None
                    and representative_frequency is not None
                    else float("inf")
                ),
                (
                    peak_frequency
                    if peak_frequency is not None
                    else float("inf")
                ),
                source.cluster.frequency,
            )

        representative_source = min(source_clusters, key=evidence_key)
        consolidated_regions.append(
            ConsolidatedFrequencyRegion(
                band_name=source_clusters[0].cluster.band_name,
                frequency=float(np.median([
                    source.cluster.frequency for source in source_clusters
                ])),
                support_count=len(session_indices),
                support_fraction=len(session_indices) / total_sessions,
                minimum_frequency=min(
                    source.cluster.minimum_frequency
                    for source in source_clusters
                ),
                maximum_frequency=max(
                    source.cluster.maximum_frequency
                    for source in source_clusters
                ),
                session_indices=session_indices,
                median_evidence=representative_source.median_evidence,
                source_clusters=source_clusters,
            )
        )
    return consolidated_regions


def build_consolidated_frequency_regions(
    diagnostics: FrequencyClusterDiagnostics,
    analysis_bands: list[AnalysisBand],
    total_sessions: int,
    config: FrequencyClusterConsolidationConfig,
) -> ConsolidatedFrequencyRegions:
    if not isinstance(diagnostics, FrequencyClusterDiagnostics):
        raise ValueError(
            "Frequency cluster diagnostics must be a "
            "FrequencyClusterDiagnostics"
        )
    return ConsolidatedFrequencyRegions(
        x=consolidate_frequency_cluster_diagnostics(
            diagnostics.x,
            analysis_bands,
            total_sessions,
            config,
        ),
        y=consolidate_frequency_cluster_diagnostics(
            diagnostics.y,
            analysis_bands,
            total_sessions,
            config,
        ),
        z=consolidate_frequency_cluster_diagnostics(
            diagnostics.z,
            analysis_bands,
            total_sessions,
            config,
        ),
    )


def validate_trusted_frequency_visualization_config(
    config: TrustedFrequencyVisualizationConfig,
) -> None:
    if not isinstance(config, TrustedFrequencyVisualizationConfig):
        raise ValueError(
            "Trusted frequency config must be a "
            "TrustedFrequencyVisualizationConfig"
        )
    values = (
        (config.min_support_fraction, "min_support_fraction"),
        (config.min_median_prominence_db, "min_median_prominence_db"),
        (config.background_weight, "background_weight"),
        (config.min_band_contrast_db, "min_band_contrast_db"),
        (config.weak_trusted_weight, "weak_trusted_weight"),
    )
    for value, name in values:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not np.isfinite(value)
        ):
            raise ValueError(
                f"Trusted frequency {name} must be a finite number"
            )
    if not 0 < config.min_support_fraction <= 1:
        raise ValueError(
            "Trusted frequency min_support_fraction must be greater than "
            "zero and at most one"
        )
    if config.min_median_prominence_db < 0:
        raise ValueError(
            "Trusted frequency min_median_prominence_db must be non-negative"
        )
    if not 0 <= config.background_weight <= 1:
        raise ValueError(
            "Trusted frequency background_weight must be between zero and one"
        )
    if config.min_band_contrast_db < 0:
        raise ValueError(
            "Trusted frequency min_band_contrast_db must be non-negative"
        )
    if not 0 <= config.weak_trusted_weight <= 1:
        raise ValueError(
            "Trusted frequency weak_trusted_weight must be between zero and one"
        )
    if config.weak_trusted_weight < config.background_weight:
        raise ValueError(
            "Trusted frequency weak_trusted_weight must not be less than "
            "background_weight"
        )


def is_trusted_frequency_cluster(
    region: ConsolidatedFrequencyRegion,
    config: TrustedFrequencyVisualizationConfig,
) -> bool:
    if not isinstance(region, ConsolidatedFrequencyRegion):
        raise ValueError(
            "Trusted frequency region must be a "
            "ConsolidatedFrequencyRegion"
        )
    validate_trusted_frequency_visualization_config(config)
    prominence_db = region.median_evidence.prominence_db
    return bool(
        region.support_fraction >= config.min_support_fraction
        and prominence_db is not None
        and prominence_db >= config.min_median_prominence_db
    )


def get_trusted_frequency_weight(
    region: ConsolidatedFrequencyRegion,
    config: TrustedFrequencyVisualizationConfig,
) -> float:
    if not isinstance(region, ConsolidatedFrequencyRegion):
        raise ValueError(
            "Trusted frequency region must be a "
            "ConsolidatedFrequencyRegion"
        )
    validate_trusted_frequency_visualization_config(config)
    if not is_trusted_frequency_cluster(region, config):
        return float(config.background_weight)
    band_contrast = region.median_evidence.band_contrast_db
    if (
        band_contrast is not None
        and band_contrast >= config.min_band_contrast_db
    ):
        return 1.0
    return float(config.weak_trusted_weight)


def build_trusted_frequency_mask(
    frequency: np.ndarray,
    regions: list[ConsolidatedFrequencyRegion],
    analysis_bands: list[AnalysisBand],
    config: TrustedFrequencyVisualizationConfig,
) -> np.ndarray:
    if not isinstance(frequency, np.ndarray):
        raise ValueError("Frequency axis must be a NumPy array")
    if frequency.ndim != 1:
        raise ValueError("Frequency axis must be one-dimensional")
    if len(frequency) == 0:
        raise ValueError("Frequency axis must not be empty")
    try:
        if not np.all(np.isfinite(frequency)):
            raise ValueError("Frequency axis must contain only finite values")
        if not np.all(np.diff(frequency) > 0):
            raise ValueError("Frequency axis must be strictly increasing")
    except TypeError as error:
        raise ValueError("Frequency axis must contain numeric values") from error
    if not isinstance(regions, list):
        raise ValueError("Consolidated frequency regions must be a list")
    if not isinstance(analysis_bands, list) or not analysis_bands:
        raise ValueError("At least one analysis band is required")
    if not all(isinstance(band, AnalysisBand) for band in analysis_bands):
        raise ValueError("Analysis bands must contain AnalysisBand values")
    validate_trusted_frequency_visualization_config(config)

    bands_by_name = {band.name: band for band in analysis_bands}
    mask = np.zeros_like(frequency, dtype=bool)
    for region in regions:
        if not isinstance(region, ConsolidatedFrequencyRegion):
            raise ValueError(
                "Consolidated frequency regions must contain "
                "ConsolidatedFrequencyRegion values"
            )
        band = bands_by_name.get(region.band_name)
        if band is None:
            raise ValueError(
                f"Frequency region band {region.band_name!r} is not configured"
            )
        if not is_trusted_frequency_cluster(region, config):
            continue
        peak_frequency = region.median_evidence.peak_frequency
        if peak_frequency is None:
            raise ValueError(
                "Trusted frequency cluster must have a Median PSD peak"
            )
        region_min = max(
            band.min_frequency,
            peak_frequency - band.frequency_tolerance_hz,
        )
        region_max = min(
            band.max_frequency,
            peak_frequency + band.frequency_tolerance_hz,
        )
        mask |= (frequency >= region_min) & (frequency <= region_max)
    return mask


def build_trusted_frequency_weight(
    frequency: np.ndarray,
    regions: list[ConsolidatedFrequencyRegion],
    analysis_bands: list[AnalysisBand],
    config: TrustedFrequencyVisualizationConfig,
) -> np.ndarray:
    trusted_mask = build_trusted_frequency_mask(
        frequency,
        regions,
        analysis_bands,
        config,
    )
    weights = np.full_like(
        frequency,
        config.background_weight,
        dtype=float,
    )
    weights[trusted_mask] = config.weak_trusted_weight

    bands_by_name = {band.name: band for band in analysis_bands}
    for region in regions:
        if get_trusted_frequency_weight(region, config) < 1.0:
            continue
        band = bands_by_name[region.band_name]
        peak_frequency = region.median_evidence.peak_frequency
        if peak_frequency is None:
            raise ValueError(
                "Trusted frequency cluster must have a Median PSD peak"
            )
        region_min = max(
            band.min_frequency,
            peak_frequency - band.frequency_tolerance_hz,
        )
        region_max = min(
            band.max_frequency,
            peak_frequency + band.frequency_tolerance_hz,
        )
        region_mask = (
            (frequency >= region_min)
            & (frequency <= region_max)
        )
        weights[region_mask] = np.maximum(weights[region_mask], 1.0)

    if np.any(weights < config.background_weight) or np.any(weights > 1):
        raise ValueError("Trusted frequency weights are outside expected bounds")
    return weights


def compute_trusted_frequency_psd(
    median_psd: np.ndarray,
    trusted_frequency_weight: np.ndarray,
) -> np.ndarray:
    if not isinstance(median_psd, np.ndarray):
        raise ValueError("Median PSD must be a NumPy array")
    if median_psd.ndim != 1:
        raise ValueError("Median PSD must be one-dimensional")
    try:
        if not np.all(np.isfinite(median_psd)):
            raise ValueError("Median PSD must contain only finite values")
        if np.any(median_psd < 0):
            raise ValueError("Median PSD must be non-negative")
    except TypeError as error:
        raise ValueError("Median PSD must contain numeric values") from error
    if not isinstance(trusted_frequency_weight, np.ndarray):
        raise ValueError("Trusted frequency weight must be a NumPy array")
    if trusted_frequency_weight.ndim != 1:
        raise ValueError("Trusted frequency weight must be one-dimensional")
    if len(trusted_frequency_weight) != len(median_psd):
        raise ValueError(
            "Trusted frequency weight length must match Median PSD length"
        )
    try:
        if not np.all(np.isfinite(trusted_frequency_weight)):
            raise ValueError(
                "Trusted frequency weight must contain only finite values"
            )
        if np.any(trusted_frequency_weight < 0) or np.any(
            trusted_frequency_weight > 1
        ):
            raise ValueError(
                "Trusted frequency weight must be between zero and one"
            )
    except TypeError as error:
        raise ValueError(
            "Trusted frequency weight must contain numeric values"
        ) from error

    result = median_psd * trusted_frequency_weight
    if np.any(result < 0) or np.any(result > median_psd):
        raise ValueError("Trusted frequency PSD is outside expected bounds")
    return result


@dataclass
class VisualizationAxis:
    frequency: np.ndarray
    median_psd: np.ndarray
    mean_psd: np.ndarray
    std_psd: np.ndarray
    stability: np.ndarray
    local_window_power_stability: np.ndarray
    local_psd_background: np.ndarray
    local_psd_contrast_db: np.ndarray
    trusted_frequency_mask: np.ndarray
    trusted_frequency_weight: np.ndarray
    trusted_frequency_psd: np.ndarray
    peak_frequencies: np.ndarray
    peak_amplitudes: np.ndarray
    peak_window_power_stability: np.ndarray
    peak_mean_session_frequencies: np.ndarray
    peak_frequency_std_hz: np.ndarray
    peak_minimum_session_frequencies: np.ndarray
    peak_maximum_session_frequencies: np.ndarray
    peak_local_noise_floor: np.ndarray
    peak_local_snr_db: np.ndarray


@dataclass
class VisualizationData:
    x: VisualizationAxis
    y: VisualizationAxis
    z: VisualizationAxis


def print_peak_candidate_diagnostics(
    diagnostics: PeakCandidateDiagnostics,
) -> None:
    for axis_name, candidates in (
        ("X", diagnostics.x),
        ("Y", diagnostics.y),
        ("Z", diagnostics.z),
    ):
        print()
        print(f"Peak candidates — {axis_name}")
        if not candidates:
            print("No candidates passed prominence/distance thresholds.")
            continue

        band_width = max(len("Band"), *(len(item.band_name) for item in candidates))
        print(
            f"{'Band':<{band_width}}  {'Freq Hz':>7}  {'Prom dB':>7}  "
            f"{'Win.Stab':>8}  {'Min.Stab':>8}  {'SNR dB':>7}  "
            f"{'σf Hz':>6}  {'σf Max':>6}  {'Range Hz':>13}  "
            f"{'Freq.Stab':>9}  Result"
        )
        for item in candidates:
            if item.accepted:
                result = "ACCEPT"
            elif item.rejection_reason == "window_power_stability":
                result = "REJECT stability"
            elif item.rejection_reason == "insufficient_local_noise_bins":
                result = "REJECT noise bins"
            else:
                result = f"REJECT {item.rejection_reason}"

            if item.frequency_stability_passed is None:
                frequency_stability = "N/A"
            elif item.frequency_stability_passed:
                frequency_stability = "PASS"
            else:
                frequency_stability = "FAIL"

            frequency_range = (
                f"{item.minimum_session_frequency:.2f}–"
                f"{item.maximum_session_frequency:.2f}"
            )
            print(
                f"{item.band_name:<{band_width}}  {item.frequency:7.2f}  "
                f"{item.prominence_db:7.2f}  "
                f"{item.window_power_stability:8.2f}  "
                f"{item.min_stability:8.2f}  {item.local_snr_db:7.2f}  "
                f"{item.frequency_std_hz:6.2f}  "
                f"{item.frequency_stability_max_std_hz:6.2f}  "
                f"{frequency_range:>13}  {frequency_stability:>9}  {result}"
            )


def find_session_peaks(
    frequency: np.ndarray,
    session_psd_stack: np.ndarray,
    analysis_bands: list[AnalysisBand],
) -> list[SessionPeak]:
    if not isinstance(frequency, np.ndarray) or frequency.ndim != 1:
        raise ValueError("Frequency must be a one-dimensional NumPy array")
    if len(frequency) == 0:
        raise ValueError("Frequency must not be empty")
    try:
        if not np.all(np.isfinite(frequency)):
            raise ValueError("Frequency must contain only finite values")
        if not np.all(np.diff(frequency) > 0):
            raise ValueError("Frequency must be strictly increasing")
    except TypeError as error:
        raise ValueError("Frequency must contain numeric values") from error

    if not isinstance(session_psd_stack, np.ndarray) or session_psd_stack.ndim != 2:
        raise ValueError("Session PSD stack must be a two-dimensional NumPy array")
    if session_psd_stack.shape[0] == 0:
        raise ValueError("Session PSD stack must contain at least one session")
    if session_psd_stack.shape[1] != len(frequency):
        raise ValueError("Session PSD stack width must match frequency length")
    try:
        if not np.all(np.isfinite(session_psd_stack)):
            raise ValueError("Session PSD stack must contain only finite values")
        if np.any(session_psd_stack < 0):
            raise ValueError("Session PSD stack must not contain negative values")
    except TypeError as error:
        raise ValueError("Session PSD stack must contain numeric values") from error

    if not isinstance(analysis_bands, list) or not analysis_bands:
        raise ValueError("At least one analysis band is required")

    session_peaks = []
    tiny = np.finfo(float).tiny
    for session_index, session_psd in enumerate(session_psd_stack, start=1):
        peaks_by_global_index: dict[int, tuple[int, SessionPeak]] = {}
        for band_index, band in enumerate(analysis_bands):
            band_mask = (
                (frequency >= band.min_frequency)
                & (frequency <= band.max_frequency)
            )
            global_indices = np.flatnonzero(band_mask)
            if len(global_indices) < 2:
                continue

            resolution = frequency[1] - frequency[0]
            distance = max(int(band.min_distance_hz / resolution), 1)
            safe_psd = np.maximum(session_psd[band_mask], tiny)
            session_psd_db = 10.0 * np.log10(safe_psd)
            local_peak_indices, properties = find_peaks(
                session_psd_db,
                prominence=band.prominence_db,
                distance=distance,
            )
            for position, local_peak_index in enumerate(local_peak_indices):
                global_peak_index = int(global_indices[local_peak_index])
                peak = SessionPeak(
                    session_index=session_index,
                    band_name=band.name,
                    frequency=float(frequency[global_peak_index]),
                    prominence_db=float(properties["prominences"][position]),
                )
                existing = peaks_by_global_index.get(global_peak_index)
                if (
                    existing is None
                    or peak.prominence_db > existing[1].prominence_db
                ):
                    peaks_by_global_index[global_peak_index] = (
                        band_index,
                        peak,
                    )

        session_peaks.extend(
            peak
            for _, peak in sorted(
                peaks_by_global_index.values(),
                key=lambda item: (item[1].frequency, item[0]),
            )
        )

    return session_peaks


def build_frequency_clusters(
    session_peaks: list[SessionPeak],
    analysis_bands: list[AnalysisBand],
    total_sessions: int,
) -> list[FrequencyCluster]:
    if not isinstance(session_peaks, list):
        raise ValueError("Session peaks must be a list")
    if not isinstance(analysis_bands, list) or not analysis_bands:
        raise ValueError("At least one analysis band is required")
    if (
        not isinstance(total_sessions, int)
        or isinstance(total_sessions, bool)
        or total_sessions <= 0
    ):
        raise ValueError("Total sessions must be a positive integer")

    band_names = {band.name for band in analysis_bands}
    for peak in session_peaks:
        if not isinstance(peak, SessionPeak):
            raise ValueError("Session peaks must contain SessionPeak values")
        if peak.band_name not in band_names:
            raise ValueError("Session peak band is not in analysis bands")
        if not 1 <= peak.session_index <= total_sessions:
            raise ValueError("Session peak index is outside total sessions")
        if not np.isfinite(peak.frequency):
            raise ValueError("Session peak frequency must be finite")
        if not np.isfinite(peak.prominence_db) or peak.prominence_db < 0:
            raise ValueError(
                "Session peak prominence must be finite and non-negative"
            )

    clusters = []
    for band in analysis_bands:
        band_peaks = sorted(
            (peak for peak in session_peaks if peak.band_name == band.name),
            key=lambda peak: (
                peak.frequency,
                peak.session_index,
                -peak.prominence_db,
            ),
        )
        representative_clusters: list[dict[int, SessionPeak]] = []
        for peak in band_peaks:
            matching_clusters = []
            for cluster_index, representatives in enumerate(
                representative_clusters
            ):
                center = float(np.median([
                    representative.frequency
                    for representative in representatives.values()
                ]))
                distance = abs(peak.frequency - center)
                if distance <= band.frequency_tolerance_hz:
                    matching_clusters.append((distance, center, cluster_index))

            if not matching_clusters:
                representative_clusters.append({peak.session_index: peak})
                continue

            _, center, cluster_index = min(matching_clusters)
            representatives = representative_clusters[cluster_index]
            existing = representatives.get(peak.session_index)
            if existing is None:
                representatives[peak.session_index] = peak
                continue

            existing_key = (
                -existing.prominence_db,
                abs(existing.frequency - center),
                existing.frequency,
            )
            peak_key = (
                -peak.prominence_db,
                abs(peak.frequency - center),
                peak.frequency,
            )
            if peak_key < existing_key:
                representatives[peak.session_index] = peak

        for representatives in representative_clusters:
            session_indices = tuple(sorted(representatives))
            representative_frequencies = np.asarray([
                representatives[index].frequency
                for index in session_indices
            ])
            support_count = len(session_indices)
            support_fraction = support_count / total_sessions
            cluster_frequency = float(np.median(representative_frequencies))
            frequency_std_hz = float(np.std(representative_frequencies))
            minimum_frequency = float(np.min(representative_frequencies))
            maximum_frequency = float(np.max(representative_frequencies))

            if not 1 <= support_count <= total_sessions:
                raise ValueError("Frequency cluster support count is invalid")
            if not 0 < support_fraction <= 1:
                raise ValueError("Frequency cluster support fraction is invalid")
            if len(session_indices) != len(set(session_indices)):
                raise ValueError("Frequency cluster session indices must be unique")
            if session_indices != tuple(sorted(session_indices)):
                raise ValueError("Frequency cluster session indices must be sorted")
            if not np.isfinite(frequency_std_hz) or frequency_std_hz < 0:
                raise ValueError("Frequency cluster standard deviation is invalid")
            if not minimum_frequency <= cluster_frequency <= maximum_frequency:
                raise ValueError("Frequency cluster center is outside its range")

            clusters.append(
                FrequencyCluster(
                    band_name=band.name,
                    frequency=cluster_frequency,
                    support_count=support_count,
                    support_fraction=support_fraction,
                    frequency_std_hz=frequency_std_hz,
                    minimum_frequency=minimum_frequency,
                    maximum_frequency=maximum_frequency,
                    session_indices=session_indices,
                )
            )

    band_order = {
        band.name: band_index
        for band_index, band in enumerate(analysis_bands)
    }
    return sorted(
        clusters,
        key=lambda cluster: (
            band_order[cluster.band_name],
            cluster.frequency,
        ),
    )


def build_session_frequency_clusters(
    aligned: AlignedPSDData,
    analysis_bands: list[AnalysisBand],
) -> FrequencyClusterResult:
    validate_aligned_psd_data(aligned)
    total_sessions = aligned.x_stack.shape[0]

    def build_axis_clusters(session_psd_stack: np.ndarray) -> list[FrequencyCluster]:
        session_peaks = find_session_peaks(
            aligned.frequency,
            session_psd_stack,
            analysis_bands,
        )
        return build_frequency_clusters(
            session_peaks,
            analysis_bands,
            total_sessions,
        )

    return FrequencyClusterResult(
        x=build_axis_clusters(aligned.x_stack),
        y=build_axis_clusters(aligned.y_stack),
        z=build_axis_clusters(aligned.z_stack),
    )


def compute_median_psd_evidence(
    frequency: np.ndarray,
    median_psd: np.ndarray,
    local_contrast_db: np.ndarray,
    cluster: FrequencyCluster,
    band: AnalysisBand,
) -> MedianPSDEvidence:
    if not isinstance(frequency, np.ndarray) or frequency.ndim != 1:
        raise ValueError("Frequency must be a one-dimensional NumPy array")
    if len(frequency) == 0:
        raise ValueError("Frequency must not be empty")
    try:
        if not np.all(np.isfinite(frequency)):
            raise ValueError("Frequency must contain only finite values")
        if not np.all(np.diff(frequency) > 0):
            raise ValueError("Frequency must be strictly increasing")
    except TypeError as error:
        raise ValueError("Frequency must contain numeric values") from error

    if not isinstance(median_psd, np.ndarray) or median_psd.ndim != 1:
        raise ValueError("Median PSD must be a one-dimensional NumPy array")
    if len(median_psd) != len(frequency):
        raise ValueError("Frequency and Median PSD lengths must match")
    try:
        if not np.all(np.isfinite(median_psd)):
            raise ValueError("Median PSD must contain only finite values")
        if np.any(median_psd < 0):
            raise ValueError("Median PSD must not contain negative values")
    except TypeError as error:
        raise ValueError("Median PSD must contain numeric values") from error

    if not isinstance(local_contrast_db, np.ndarray) or local_contrast_db.ndim != 1:
        raise ValueError("Local PSD contrast must be a one-dimensional NumPy array")
    if len(local_contrast_db) != len(frequency):
        raise ValueError("Frequency and Local PSD contrast lengths must match")
    try:
        finite_contrast = local_contrast_db[~np.isnan(local_contrast_db)]
        if not np.all(np.isfinite(finite_contrast)):
            raise ValueError(
                "Local PSD contrast must contain only finite values or NaN"
            )
    except TypeError as error:
        raise ValueError("Local PSD contrast must contain numeric values") from error

    if not isinstance(cluster, FrequencyCluster):
        raise ValueError("Cluster must be a FrequencyCluster")
    cluster_values = (
        cluster.frequency,
        cluster.minimum_frequency,
        cluster.maximum_frequency,
    )
    if not all(np.isfinite(value) for value in cluster_values):
        raise ValueError("Frequency cluster frequencies must be finite")
    if not (
        cluster.minimum_frequency
        <= cluster.frequency
        <= cluster.maximum_frequency
    ):
        raise ValueError("Frequency cluster center must be inside its range")
    if cluster.band_name != band.name:
        raise ValueError("Frequency cluster and analysis band names must match")

    no_evidence = MedianPSDEvidence(
        peak_frequency=None,
        peak_psd=None,
        prominence_db=None,
        local_contrast_db=None,
        band_contrast_db=None,
        passed_prominence=None,
    )
    search_min = max(
        band.min_frequency,
        cluster.minimum_frequency - band.frequency_tolerance_hz,
    )
    search_max = min(
        band.max_frequency,
        cluster.maximum_frequency + band.frequency_tolerance_hz,
    )
    band_mask = (
        (frequency >= band.min_frequency)
        & (frequency <= band.max_frequency)
    )
    band_global_indices = np.flatnonzero(band_mask)
    if len(band_global_indices) < 3:
        return no_evidence

    safe_median = np.maximum(median_psd, np.finfo(float).tiny)
    band_median_db = 10.0 * np.log10(safe_median[band_mask])
    local_peak_indices, properties = find_peaks(
        band_median_db,
        prominence=0,
    )
    if len(local_peak_indices) == 0:
        return no_evidence

    peak_candidates = []
    for position, local_peak_index in enumerate(local_peak_indices):
        global_peak_index = int(band_global_indices[local_peak_index])
        peak_frequency = float(frequency[global_peak_index])
        if not search_min <= peak_frequency <= search_max:
            continue
        prominence_db = float(properties["prominences"][position])
        peak_candidates.append((
            -prominence_db,
            abs(peak_frequency - cluster.frequency),
            peak_frequency,
            global_peak_index,
            prominence_db,
        ))

    if not peak_candidates:
        return no_evidence

    (
        _,
        _,
        peak_frequency,
        global_peak_index,
        prominence_db,
    ) = min(peak_candidates)
    contrast_value = local_contrast_db[global_peak_index]
    local_contrast = (
        float(contrast_value)
        if np.isfinite(contrast_value)
        else None
    )
    peak_psd = float(median_psd[global_peak_index])
    band_reference = float(np.median(median_psd[band_mask]))
    tiny = np.finfo(float).tiny
    safe_peak_psd = max(peak_psd, tiny)
    safe_band_reference = max(band_reference, tiny)
    band_contrast_db = float(
        10.0 * np.log10(safe_peak_psd / safe_band_reference)
    )
    if not np.isfinite(band_contrast_db):
        raise ValueError("Band-level Median PSD contrast must be finite")
    return MedianPSDEvidence(
        peak_frequency=peak_frequency,
        peak_psd=peak_psd,
        prominence_db=prominence_db,
        local_contrast_db=local_contrast,
        band_contrast_db=band_contrast_db,
        passed_prominence=bool(prominence_db >= band.prominence_db),
    )


def build_frequency_cluster_diagnostics(
    frequency: np.ndarray,
    median_psd: np.ndarray,
    clusters: list[FrequencyCluster],
    analysis_bands: list[AnalysisBand],
) -> list[FrequencyClusterDiagnostic]:
    if not isinstance(clusters, list):
        raise ValueError("Frequency clusters must be a list")
    if not isinstance(analysis_bands, list) or not analysis_bands:
        raise ValueError("At least one analysis band is required")

    local_background = compute_local_psd_background(
        frequency,
        median_psd,
        analysis_bands,
    )
    local_contrast_db = compute_local_psd_contrast_db(
        median_psd,
        local_background,
    )
    diagnostics = []
    for cluster in clusters:
        if not isinstance(cluster, FrequencyCluster):
            raise ValueError(
                "Frequency clusters must contain FrequencyCluster values"
            )
        band = next(
            (
                candidate_band
                for candidate_band in analysis_bands
                if candidate_band.name == cluster.band_name
            ),
            None,
        )
        if band is None:
            raise ValueError(
                f"Frequency cluster band {cluster.band_name!r} is not configured"
            )
        diagnostics.append(
            FrequencyClusterDiagnostic(
                cluster=cluster,
                median_evidence=compute_median_psd_evidence(
                    frequency,
                    median_psd,
                    local_contrast_db,
                    cluster,
                    band,
                ),
            )
        )
    return diagnostics


def build_session_frequency_cluster_diagnostics(
    statistics: StatisticsResult,
    clusters: FrequencyClusterResult,
    analysis_bands: list[AnalysisBand],
    frequency: np.ndarray,
) -> FrequencyClusterDiagnostics:
    return FrequencyClusterDiagnostics(
        x=build_frequency_cluster_diagnostics(
            frequency,
            statistics.x.median,
            clusters.x,
            analysis_bands,
        ),
        y=build_frequency_cluster_diagnostics(
            frequency,
            statistics.y.median,
            clusters.y,
            analysis_bands,
        ),
        z=build_frequency_cluster_diagnostics(
            frequency,
            statistics.z.median,
            clusters.z,
            analysis_bands,
        ),
    )


def print_session_frequency_clusters(
    diagnostics: FrequencyClusterDiagnostics,
    analysis_bands: list[AnalysisBand],
    total_sessions: int,
) -> None:
    band_order = {
        band.name: band_index
        for band_index, band in enumerate(analysis_bands)
    }
    for axis_name, axis_diagnostics in (
        ("X", diagnostics.x),
        ("Y", diagnostics.y),
        ("Z", diagnostics.z),
    ):
        print()
        print(f"Session frequency clusters — {axis_name}")
        if not axis_diagnostics:
            print("No session frequency clusters.")
            continue

        sorted_diagnostics = sorted(
            axis_diagnostics,
            key=lambda diagnostic: (
                band_order[diagnostic.cluster.band_name],
                diagnostic.cluster.frequency,
            ),
        )
        band_width = max(
            len("Band"),
            *(
                len(diagnostic.cluster.band_name)
                for diagnostic in sorted_diagnostics
            ),
        )
        print(
            f"{'Band':<{band_width}}  {'Freq Hz':>7}  {'Support':>7}  "
            f"{'Support %':>9}  {'σf Hz':>6}  {'Range Hz':>13}  "
            f"{'Med.Freq':>8}  {'Med.Prom':>8}  {'Med.Contr':>9}  "
            f"{'Band.Contr':>10}  {'Med.Pass':>8}  Sessions"
        )
        for diagnostic in sorted_diagnostics:
            cluster = diagnostic.cluster
            evidence = diagnostic.median_evidence
            frequency_range = (
                f"{cluster.minimum_frequency:.2f}–"
                f"{cluster.maximum_frequency:.2f}"
            )
            session_indices = ",".join(
                str(index) for index in cluster.session_indices
            )
            median_frequency = (
                f"{evidence.peak_frequency:.2f}"
                if evidence.peak_frequency is not None
                else "N/A"
            )
            median_prominence = (
                f"{evidence.prominence_db:.2f}"
                if evidence.prominence_db is not None
                else "N/A"
            )
            median_contrast = (
                f"{evidence.local_contrast_db:.2f}"
                if evidence.local_contrast_db is not None
                else "N/A"
            )
            band_contrast = (
                f"{evidence.band_contrast_db:.2f}"
                if evidence.band_contrast_db is not None
                else "N/A"
            )
            if evidence.passed_prominence is None:
                median_pass = "N/A"
            elif evidence.passed_prominence:
                median_pass = "PASS"
            else:
                median_pass = "FAIL"
            print(
                f"{cluster.band_name:<{band_width}}  "
                f"{cluster.frequency:7.2f}  "
                f"{cluster.support_count:>3}/{total_sessions:<3}  "
                f"{cluster.support_fraction * 100:9.1f}  "
                f"{cluster.frequency_std_hz:6.2f}  "
                f"{frequency_range:>13}  {median_frequency:>8}  "
                f"{median_prominence:>8}  {median_contrast:>9}  "
                f"{band_contrast:>10}  {median_pass:>8}  {session_indices}"
            )


def print_consolidated_frequency_regions(
    regions: ConsolidatedFrequencyRegions,
    total_sessions: int,
) -> None:
    for axis_name, axis_regions in (
        ("X", regions.x),
        ("Y", regions.y),
        ("Z", regions.z),
    ):
        print()
        print(f"Consolidated frequency regions — {axis_name}")
        if not axis_regions:
            print("No consolidated frequency regions.")
            continue
        band_width = max(
            len("Band"),
            *(len(region.band_name) for region in axis_regions),
        )
        print(
            f"{'Band':<{band_width}}  {'Freq Hz':>7}  {'Support':>7}  "
            f"{'Med.Freq':>8}  {'Med.Prom':>8}  {'Med.Contr':>9}  "
            f"{'Band.Contr':>10}  {'Range Hz':>13}  {'Sources':>7}"
        )
        for region in axis_regions:
            evidence = region.median_evidence
            median_frequency = (
                f"{evidence.peak_frequency:.2f}"
                if evidence.peak_frequency is not None
                else "N/A"
            )
            median_prominence = (
                f"{evidence.prominence_db:.2f}"
                if evidence.prominence_db is not None
                else "N/A"
            )
            median_contrast = (
                f"{evidence.local_contrast_db:.2f}"
                if evidence.local_contrast_db is not None
                else "N/A"
            )
            band_contrast = (
                f"{evidence.band_contrast_db:.2f}"
                if evidence.band_contrast_db is not None
                else "N/A"
            )
            frequency_range = (
                f"{region.minimum_frequency:.2f}–"
                f"{region.maximum_frequency:.2f}"
            )
            print(
                f"{region.band_name:<{band_width}}  "
                f"{region.frequency:7.2f}  "
                f"{region.support_count:>3}/{total_sessions:<3}  "
                f"{median_frequency:>8}  {median_prominence:>8}  "
                f"{median_contrast:>9}  {band_contrast:>10}  "
                f"{frequency_range:>13}  "
                f"{len(region.source_clusters):7d}"
            )


def print_trusted_frequency_regions(
    regions: ConsolidatedFrequencyRegions,
    config: TrustedFrequencyVisualizationConfig,
    total_sessions: int,
) -> None:
    validate_trusted_frequency_visualization_config(config)
    for axis_name, axis_regions in (
        ("X", regions.x),
        ("Y", regions.y),
        ("Z", regions.z),
    ):
        print()
        print(f"Trusted frequency regions — {axis_name}")
        trusted_regions = [
            region
            for region in axis_regions
            if is_trusted_frequency_cluster(region, config)
        ]
        if not trusted_regions:
            print("No trusted frequency regions.")
            continue
        band_width = max(
            len("Band"),
            *(len(region.band_name) for region in trusted_regions),
        )
        print(
            f"{'Band':<{band_width}}  {'Freq Hz':>7}  {'Support':>7}  "
            f"{'Med.Freq':>8}  {'Med.Prom':>8}  {'Med.Contr':>9}  "
            f"{'Band.Contr':>10}  {'Weight':>6}  {'Range Hz':>13}"
        )
        for region in trusted_regions:
            evidence = region.median_evidence
            median_contrast = (
                f"{evidence.local_contrast_db:.2f}"
                if evidence.local_contrast_db is not None
                else "N/A"
            )
            band_contrast = (
                f"{evidence.band_contrast_db:.2f}"
                if evidence.band_contrast_db is not None
                else "N/A"
            )
            weight = get_trusted_frequency_weight(region, config)
            print(
                f"{region.band_name:<{band_width}}  "
                f"{region.frequency:7.2f}  "
                f"{region.support_count:>3}/{total_sessions:<3}  "
                f"{evidence.peak_frequency:8.2f}  "
                f"{evidence.prominence_db:8.2f}  "
                f"{median_contrast:>9}  "
                f"{band_contrast:>10}  "
                f"{weight:6.2f}  "
                f"{region.minimum_frequency:.2f}–"
                f"{region.maximum_frequency:.2f}"
            )


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
    peak_frequency_std_hz,
    local_snr_db,
) -> None:
    if len(peak_frequencies) != len(peak_values):
        raise ValueError("Peak frequency and value arrays must have equal lengths")
    if len(peak_frequencies) != len(peak_frequency_std_hz):
        raise ValueError(
            "Peak frequency and frequency spread arrays must have equal lengths"
        )
    if len(peak_frequencies) != len(local_snr_db):
        raise ValueError(
            "Peak frequency and local SNR arrays must have equal lengths"
        )

    for index in range(len(peak_frequencies)):
        frequency = peak_frequencies[index]
        value = peak_values[index]
        ax.annotate(
            f"{frequency:.1f} Hz\n"
            f"σf {peak_frequency_std_hz[index]:.2f} Hz\n"
            f"SNR {local_snr_db[index]:.1f} dB",
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
    aligned: AlignedPSDData,
    analysis_bands: list[AnalysisBand],
    consolidated_frequency_regions: ConsolidatedFrequencyRegions,
    trusted_frequency_config: TrustedFrequencyVisualizationConfig,
) -> VisualizationData:
    validate_aligned_psd_data(aligned)
    frequency = aligned.frequency
    if not isinstance(frequency, np.ndarray):
        raise ValueError("Frequency axis must be a NumPy array")
    if frequency.ndim != 1:
        raise ValueError("Frequency axis must be one-dimensional")
    if len(frequency) < 2:
        raise ValueError("Frequency axis must contain at least two points")
    try:
        if not np.all(np.isfinite(frequency)):
            raise ValueError("Frequency axis must contain only finite values")
        if not np.all(np.diff(frequency) > 0):
            raise ValueError("Frequency axis must be strictly increasing")
    except TypeError as error:
        raise ValueError("Frequency axis must contain numeric values") from error

    def build_axis(
        axis_name: str,
        axis_statistics: AxisStatistics,
        axis_peaks: AxisPeaks,
        session_psd_stack: np.ndarray,
        axis_frequency_regions: list[ConsolidatedFrequencyRegion],
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

        peak_count = len(axis_peaks.frequencies)
        diagnostic_arrays = {
            "window power stability": (
                axis_peaks.diagnostics.window_power_stability
            ),
            "mean session frequencies": (
                axis_peaks.diagnostics.mean_session_frequencies
            ),
            "frequency standard deviation": (
                axis_peaks.diagnostics.frequency_std_hz
            ),
            "minimum session frequencies": (
                axis_peaks.diagnostics.minimum_session_frequencies
            ),
            "maximum session frequencies": (
                axis_peaks.diagnostics.maximum_session_frequencies
            ),
            "local noise floor": (
                axis_peaks.diagnostics.local_noise_floor
            ),
            "local SNR": axis_peaks.diagnostics.local_snr_db,
        }
        for diagnostic_name, diagnostic_array in diagnostic_arrays.items():
            if len(diagnostic_array) != peak_count:
                raise ValueError(
                    f"Peak {diagnostic_name} length does not match "
                    f"peak frequency length for {axis_name}"
                )

        local_psd_background = compute_local_psd_background(
            frequency,
            axis_statistics.median,
            analysis_bands,
        )
        local_psd_contrast_db = compute_local_psd_contrast_db(
            axis_statistics.median,
            local_psd_background,
        )
        local_window_power_stability = compute_local_window_power_stability(
            frequency,
            session_psd_stack,
            analysis_bands,
        )
        trusted_frequency_mask = build_trusted_frequency_mask(
            frequency,
            axis_frequency_regions,
            analysis_bands,
            trusted_frequency_config,
        )
        trusted_frequency_weight = build_trusted_frequency_weight(
            frequency,
            axis_frequency_regions,
            analysis_bands,
            trusted_frequency_config,
        )
        trusted_frequency_psd = compute_trusted_frequency_psd(
            axis_statistics.median,
            trusted_frequency_weight,
        )
        if len(local_psd_background) != expected_length:
            raise ValueError(
                f"Local PSD background length does not match frequency length "
                f"for {axis_name}"
            )
        if len(local_psd_contrast_db) != expected_length:
            raise ValueError(
                f"Local PSD contrast length does not match frequency length "
                f"for {axis_name}"
            )
        if len(local_window_power_stability) != expected_length:
            raise ValueError(
                f"Local window power stability length does not match frequency "
                f"length for {axis_name}"
            )
        if len(trusted_frequency_mask) != expected_length:
            raise ValueError(
                f"Trusted frequency mask length does not match frequency "
                f"length for {axis_name}"
            )
        if len(trusted_frequency_weight) != expected_length:
            raise ValueError(
                f"Trusted frequency weight length does not match frequency "
                f"length for {axis_name}"
            )
        if len(trusted_frequency_psd) != expected_length:
            raise ValueError(
                f"Trusted frequency PSD length does not match frequency "
                f"length for {axis_name}"
            )
        return VisualizationAxis(
            frequency=frequency,
            median_psd=axis_statistics.median,
            mean_psd=axis_statistics.mean,
            std_psd=axis_statistics.std,
            stability=axis_statistics.stability,
            local_window_power_stability=local_window_power_stability,
            local_psd_background=local_psd_background,
            local_psd_contrast_db=local_psd_contrast_db,
            trusted_frequency_mask=trusted_frequency_mask,
            trusted_frequency_weight=trusted_frequency_weight,
            trusted_frequency_psd=trusted_frequency_psd,
            peak_frequencies=axis_peaks.frequencies,
            peak_amplitudes=axis_peaks.amplitudes,
            peak_window_power_stability=(
                axis_peaks.diagnostics.window_power_stability
            ),
            peak_mean_session_frequencies=(
                axis_peaks.diagnostics.mean_session_frequencies
            ),
            peak_frequency_std_hz=(
                axis_peaks.diagnostics.frequency_std_hz
            ),
            peak_minimum_session_frequencies=(
                axis_peaks.diagnostics.minimum_session_frequencies
            ),
            peak_maximum_session_frequencies=(
                axis_peaks.diagnostics.maximum_session_frequencies
            ),
            peak_local_noise_floor=(
                axis_peaks.diagnostics.local_noise_floor
            ),
            peak_local_snr_db=axis_peaks.diagnostics.local_snr_db,
        )

    return VisualizationData(
        x=build_axis(
            "X",
            statistics.x,
            peaks.x,
            aligned.x_stack,
            consolidated_frequency_regions.x,
        ),
        y=build_axis(
            "Y",
            statistics.y,
            peaks.y,
            aligned.y_stack,
            consolidated_frequency_regions.y,
        ),
        z=build_axis(
            "Z",
            statistics.z,
            peaks.z,
            aligned.z_stack,
            consolidated_frequency_regions.z,
        ),
    )


def _find_axis_peaks_by_bands(
    freq: np.ndarray,
    median: np.ndarray,
    stability: np.ndarray,
    session_psd_stack: np.ndarray,
    analysis_bands: list[AnalysisBand],
    candidate_diagnostics: list[PeakCandidateDiagnostic],
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
            diagnostics=PeakDiagnostics(
                window_power_stability=np.array([]),
                mean_session_frequencies=np.array([]),
                frequency_std_hz=np.array([]),
                minimum_session_frequencies=np.array([]),
                maximum_session_frequencies=np.array([]),
                local_noise_floor=np.array([]),
                local_snr_db=np.array([]),
            ),
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
    diagnostics_by_index = {}
    candidate_diagnostics_by_index = {}
    property_dtypes = {}

    def store_candidate_diagnostic(
        peak_index: int,
        diagnostic: PeakCandidateDiagnostic,
    ) -> None:
        existing = candidate_diagnostics_by_index.get(peak_index)
        if (
            existing is None
            or (diagnostic.accepted and not existing.accepted)
            or (
                diagnostic.accepted == existing.accepted
                and diagnostic.prominence_db > existing.prominence_db
            )
        ):
            candidate_diagnostics_by_index[peak_index] = diagnostic

    for band in analysis_bands:
        band_mask = (
            (freq >= band.min_frequency)
            & (freq <= band.max_frequency)
        )
        global_indices = np.flatnonzero(band_mask)
        if len(global_indices) < 2:
            continue

        band_median_db = median_db[band_mask]
        distance = max(int(band.min_distance_hz / resolution), 1)
        local_peak_indices, properties = find_peaks(
            band_median_db,
            prominence=band.prominence_db,
            distance=distance,
        )
        property_dtypes.update(
            (name, values.dtype) for name, values in properties.items()
        )

        for position, local_peak_index in enumerate(local_peak_indices):
            global_peak_index = global_indices[local_peak_index]
            candidate_frequency = freq[global_peak_index]
            candidate_prominence_db = float(properties["prominences"][position])
            noise_mask = build_local_noise_mask(
                freq,
                candidate_frequency,
                band.frequency_tolerance_hz,
                band.noise_window_hz,
                band.min_frequency,
                band.max_frequency,
            )
            try:
                local_noise_floor, local_snr_db = compute_local_snr_db(
                    median,
                    global_peak_index,
                    noise_mask,
                )
            except ValueError:
                if np.count_nonzero(noise_mask) < 3:
                    unavailable = float("nan")
                    store_candidate_diagnostic(
                        global_peak_index,
                        PeakCandidateDiagnostic(
                            band_name=band.name,
                            min_stability=band.min_stability,
                            frequency_stability_max_std_hz=(
                                band.frequency_stability_max_std_hz
                            ),
                            frequency=float(candidate_frequency),
                            prominence_db=candidate_prominence_db,
                            window_power_stability=unavailable,
                            local_noise_floor=unavailable,
                            local_snr_db=unavailable,
                            mean_session_frequency=unavailable,
                            frequency_std_hz=unavailable,
                            frequency_stability_passed=None,
                            minimum_session_frequency=unavailable,
                            maximum_session_frequency=unavailable,
                            accepted=False,
                            rejection_reason="insufficient_local_noise_bins",
                        ),
                    )
                    continue
                raise
            window_mask = build_frequency_window_mask(
                freq,
                candidate_frequency,
                band.frequency_tolerance_hz,
            )
            window_powers = []
            session_peak_frequencies = []
            for session_psd in session_psd_stack:
                window_powers.append(
                    compute_window_power(freq, session_psd, window_mask)
                )
                session_peak_frequency, _ = find_session_peak_in_window(
                    freq,
                    session_psd,
                    window_mask,
                )
                session_peak_frequencies.append(session_peak_frequency)

            window_powers = np.asarray(window_powers)
            session_peak_frequencies = np.asarray(session_peak_frequencies)
            power_mean = np.mean(window_powers)
            power_std = np.std(window_powers)
            window_power_stability = np.divide(
                power_mean,
                power_std,
                out=np.array(0.0),
                where=power_std != 0,
            )
            mean_session_frequency = float(np.mean(session_peak_frequencies))
            frequency_std_hz = float(np.std(session_peak_frequencies))
            frequency_stability_passed = (
                frequency_std_hz
                <= band.frequency_stability_max_std_hz
            )
            minimum_session_frequency = float(np.min(session_peak_frequencies))
            maximum_session_frequency = float(np.max(session_peak_frequencies))
            accepted_peak_diagnostics = {
                "window_power_stability": float(window_power_stability),
                "mean_session_frequency": mean_session_frequency,
                "frequency_std_hz": frequency_std_hz,
                "minimum_session_frequency": minimum_session_frequency,
                "maximum_session_frequency": maximum_session_frequency,
                "local_noise_floor": local_noise_floor,
                "local_snr_db": local_snr_db,
            }

            rejected_for_stability = window_power_stability < band.min_stability
            accepted = not rejected_for_stability
            store_candidate_diagnostic(
                global_peak_index,
                PeakCandidateDiagnostic(
                    band_name=band.name,
                    min_stability=band.min_stability,
                    frequency_stability_max_std_hz=(
                        band.frequency_stability_max_std_hz
                    ),
                    frequency=float(candidate_frequency),
                    prominence_db=candidate_prominence_db,
                    window_power_stability=float(window_power_stability),
                    local_noise_floor=local_noise_floor,
                    local_snr_db=local_snr_db,
                    mean_session_frequency=mean_session_frequency,
                    frequency_std_hz=frequency_std_hz,
                    frequency_stability_passed=bool(
                        frequency_stability_passed
                    ),
                    minimum_session_frequency=minimum_session_frequency,
                    maximum_session_frequency=maximum_session_frequency,
                    accepted=bool(accepted),
                    rejection_reason=(
                        None if accepted else "window_power_stability"
                    ),
                ),
            )

            if rejected_for_stability:
                continue

            peak_properties = {
                name: values[position]
                for name, values in properties.items()
            }
            for name in ("left_bases", "right_bases"):
                if name in peak_properties:
                    peak_properties[name] = global_indices[peak_properties[name]]

            existing_properties = peaks_by_index.get(global_peak_index)
            existing_prominence_db = (
                existing_properties["prominences"]
                if existing_properties is not None
                else None
            )
            if (
                existing_prominence_db is None
                or peak_properties["prominences"] > existing_prominence_db
            ):
                peaks_by_index[global_peak_index] = peak_properties
                diagnostics_by_index[global_peak_index] = accepted_peak_diagnostics

    peak_indices = np.array(sorted(peaks_by_index), dtype=int)
    merged_properties = {
        name: np.asarray([
            peaks_by_index[peak_index][name]
            for peak_index in peak_indices
        ], dtype=dtype)
        for name, dtype in property_dtypes.items()
    }

    peak_frequencies = freq[peak_indices]
    candidate_diagnostics.extend(
        candidate_diagnostics_by_index[index]
        for index in sorted(candidate_diagnostics_by_index)
    )
    return AxisPeaks(
        frequencies=peak_frequencies,
        amplitudes=median[peak_indices],
        diagnostics=PeakDiagnostics(
            window_power_stability=np.asarray([
                diagnostics_by_index[index]["window_power_stability"]
                for index in peak_indices
            ]),
            mean_session_frequencies=np.asarray([
                diagnostics_by_index[index]["mean_session_frequency"]
                for index in peak_indices
            ]),
            frequency_std_hz=np.asarray([
                diagnostics_by_index[index]["frequency_std_hz"]
                for index in peak_indices
            ]),
            minimum_session_frequencies=np.asarray([
                diagnostics_by_index[index]["minimum_session_frequency"]
                for index in peak_indices
            ]),
            maximum_session_frequencies=np.asarray([
                diagnostics_by_index[index]["maximum_session_frequency"]
                for index in peak_indices
            ]),
            local_noise_floor=np.asarray([
                diagnostics_by_index[index]["local_noise_floor"]
                for index in peak_indices
            ]),
            local_snr_db=np.asarray([
                diagnostics_by_index[index]["local_snr_db"]
                for index in peak_indices
            ]),
        ),
        properties=merged_properties,
    )


def find_psd_peaks(
    statistics: StatisticsResult,
    aligned: AlignedPSDData,
    analysis_bands: list[AnalysisBand],
    candidate_diagnostics: PeakCandidateDiagnostics | None = None,
) -> PeakResult:
    validate_aligned_psd_data(aligned)
    if not analysis_bands:
        raise ValueError("At least one analysis band is required")

    expected_length = len(aligned.frequency)
    for axis_name in ("x", "y", "z"):
        axis_statistics = getattr(statistics, axis_name)
        for statistic_name in ("median", "mean", "std", "stability"):
            statistic = getattr(axis_statistics, statistic_name)
            if len(statistic) != expected_length:
                raise ValueError(
                    f"Frequency axis length does not match "
                    f"{axis_name.upper()} {statistic_name} PSD length"
                )

    if candidate_diagnostics is None:
        candidate_diagnostics = PeakCandidateDiagnostics(x=[], y=[], z=[])

    return PeakResult(
        x=_find_axis_peaks_by_bands(
            aligned.frequency,
            statistics.x.median,
            statistics.x.stability,
            aligned.x_stack,
            analysis_bands,
            candidate_diagnostics.x,
        ),
        y=_find_axis_peaks_by_bands(
            aligned.frequency,
            statistics.y.median,
            statistics.y.stability,
            aligned.y_stack,
            analysis_bands,
            candidate_diagnostics.y,
        ),
        z=_find_axis_peaks_by_bands(
            aligned.frequency,
            statistics.z.median,
            statistics.z.stability,
            aligned.z_stack,
            analysis_bands,
            candidate_diagnostics.z,
        ),
    )


def compute_statistics(
    aligned: AlignedPSDData,
) -> StatisticsResult:
    validate_aligned_psd_data(aligned)

    def compute_axis_statistics(stack: np.ndarray) -> AxisStatistics:
        median = np.median(stack, axis=0)
        mean = np.mean(stack, axis=0)
        std = np.std(stack, axis=0)
        stability = np.divide(
            mean,
            std,
            out=np.zeros_like(mean),
            where=std != 0,
        )

        return AxisStatistics(
            median=median,
            mean=mean,
            std=std,
            stability=stability,
        )

    return StatisticsResult(
        x=compute_axis_statistics(aligned.x_stack),
        y=compute_axis_statistics(aligned.y_stack),
        z=compute_axis_statistics(aligned.z_stack),
    )
# ==========================================================

try:
    cli_arguments = parse_cli_arguments()
    config = build_effective_config_from_cli(CONFIG_PATH, cli_arguments)
except FileNotFoundError:
    raise SystemExit(f"Configuration file not found: {CONFIG_PATH}")
except tomllib.TOMLDecodeError as error:
    raise SystemExit(f"Invalid TOML configuration: {error}")
except (KeyError, TypeError, ValueError) as error:
    raise SystemExit(f"Invalid configuration: {error}")

total_runs = cli_arguments.repeat

print(f"ODR: {config.sensor.odr_hz:g} Hz")
print(f"Packets/session: {config.session.packets_per_session}")
print(f"Target sessions: {config.session.min_recommended_sessions}")
print(
    "Frequency tolerance: "
    f"{resolve_frequency_tolerance_hz(config.frequency_clustering, config.sensor.odr_hz):.2f} "
    "Hz"
)

analysis_bands = config.analysis_bands
analysis_min_frequency, analysis_max_frequency = (
    get_analysis_frequency_limits(analysis_bands)
)

ser = serial.Serial(
    config.serial.port,
    config.serial.baud,
    timeout=config.serial.timeout_seconds,
)

send_adxl355_odr_command(ser, config)

print("Connected:", config.serial.port)
print(f"ADXL355 ODR command sent: {config.sensor.odr_hz:g} Hz")


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

def run_measurement(
    run_number: int,
    total_runs: int,
    show_figures: bool,
) -> bool:
    if total_runs == 1:
        print()
        input("Press ENTER to start recording...")
    else:
        print()
        print("=" * 60)
        print(f"Starting run {run_number}/{total_runs}")
        print("=" * 60)

    run_started_at = datetime.now()
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
    raw_packets = {
        "X": [],
        "Y": [],
        "Z": [],
    }

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

        raw_packets["X"].append(x)
        raw_packets["Y"].append(y)
        raw_packets["Z"].append(z)

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

        if len(sessions) >= config.session.min_recommended_sessions:
            print()
            print(f"Target session count reached: {len(sessions)}")
            break

        if (
            stop_requested
            and session_packets == config.session.packets_per_session
        ):
            break

    print()

    statistics: StatisticsResult | None = None
    peaks: PeakResult | None = None
    visualization_data: VisualizationData | None = None
    frequency_cluster_diagnostics: FrequencyClusterDiagnostics | None = None
    consolidated_frequency_regions: ConsolidatedFrequencyRegions | None = None
    run_result_paths: RunResultPaths | None = None

    if sessions:
        run_is_complete = (
            len(sessions) >= config.session.min_recommended_sessions
        )
        measured_fs = float(np.mean(fs_list))
        duration_seconds = sum(session.samples for session in sessions) / measured_fs
        run_result_paths = build_run_result_paths(
            run_started_at,
            config.sensor.odr_hz,
            run_number,
        )
        raw_packet_count = None
        raw_samples_per_packet = None
        if run_is_complete:
            raw_measurement = RawMeasurement(
                x=np.stack(raw_packets["X"]),
                y=np.stack(raw_packets["Y"]),
                z=np.stack(raw_packets["Z"]),
                packet_fs_hz=np.asarray(fs_list, dtype=float),
                created_at=run_started_at.isoformat(timespec="seconds"),
                requested_odr_hz=config.sensor.odr_hz,
                packets_per_session=config.session.packets_per_session,
                target_sessions=config.session.min_recommended_sessions,
            )
            save_raw_measurement(run_result_paths.raw, raw_measurement)
            raw_packet_count = raw_measurement.x.shape[0]
            raw_samples_per_packet = raw_measurement.x.shape[1]
        initialize_run_log(
            run_result_paths,
            run_started_at,
            config,
            run_number,
            total_runs,
            len(fs_list),
            duration_seconds,
            measured_fs,
            raw_packet_count,
            raw_samples_per_packet,
        )
        aligned_psd = build_aligned_psd_data(sessions)
        statistics = compute_statistics(aligned_psd)
        candidate_diagnostics = PeakCandidateDiagnostics(x=[], y=[], z=[])
        peaks = find_psd_peaks(
            statistics,
            aligned_psd,
            analysis_bands,
            candidate_diagnostics,
        )
        append_run_diagnostics(
            run_result_paths.log,
            print_peak_candidate_diagnostics,
            candidate_diagnostics,
        )
        frequency_clusters = build_session_frequency_clusters(
            aligned_psd,
            analysis_bands,
        )
        frequency_cluster_diagnostics = (
            build_session_frequency_cluster_diagnostics(
                statistics,
                frequency_clusters,
                analysis_bands,
                aligned_psd.frequency,
            )
        )
        append_run_diagnostics(
            run_result_paths.log,
            print_session_frequency_clusters,
            frequency_cluster_diagnostics,
            analysis_bands,
            aligned_psd.x_stack.shape[0],
        )
        consolidated_frequency_regions = build_consolidated_frequency_regions(
            frequency_cluster_diagnostics,
            analysis_bands,
            aligned_psd.x_stack.shape[0],
            config.frequency_cluster_consolidation,
        )
        append_run_diagnostics(
            run_result_paths.log,
            print_consolidated_frequency_regions,
            consolidated_frequency_regions,
            aligned_psd.x_stack.shape[0],
        )
        append_run_diagnostics(
            run_result_paths.log,
            print_trusted_frequency_regions,
            consolidated_frequency_regions,
            config.visualization.trusted_frequency,
            aligned_psd.x_stack.shape[0],
        )
        visualization_data = build_visualization_data(
            statistics,
            peaks,
            aligned_psd,
            analysis_bands,
            consolidated_frequency_regions,
            config.visualization.trusted_frequency,
        )

    if not sessions:
        print("No completed sessions available for analysis.")
        return True

    if len(sessions) < config.session.min_recommended_sessions:
        warning = (
            f"Warning: only {len(sessions)} completed session(s); "
            f"at least {config.session.min_recommended_sessions} are recommended."
        )
        print(warning)
        if run_result_paths is not None:
            with run_result_paths.log.open(
                "a",
                encoding="utf-8",
                newline="\n",
            ) as run_log:
                run_log.write(f"{warning}\n")

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
        psd_axis.plot(
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
            axis_data.peak_frequency_std_hz,
            axis_data.peak_local_snr_db,
        )
        psd_axis.set_title(f"{axis_name} axis — Median PSD")
        psd_axis.set_ylabel("PSD [g²/Hz]")
        psd_axis.set_xlim(analysis_min_frequency, analysis_max_frequency)
        psd_axis.grid(True, alpha=0.25, linewidth=0.6)
        if axis_index == 0:
            psd_axis.legend()

        stability_axis.plot(
            axis_data.frequency,
            axis_data.stability,
            color=COLORS[axis_name],
            label="Bin stability",
        )
        stability_axis.plot(
            axis_data.frequency,
            axis_data.local_window_power_stability,
            color="tab:purple",
            linewidth=1.0,
            alpha=0.75,
            label="Local window power stability",
        )
        for band_index, band in enumerate(analysis_bands):
            stability_axis.hlines(
                band.min_stability,
                band.min_frequency,
                band.max_frequency,
                linestyle="--",
                linewidth=1.0,
                alpha=0.6,
                label=(
                    "Minimum window-power stability"
                    if band_index == 0
                    else None
                ),
            )
        stability_axis.scatter(
            axis_data.peak_frequencies,
            axis_data.peak_window_power_stability,
            color=COLORS[axis_name],
            marker="x",
            label="Stable peaks — window power",
        )
        stability_axis.set_title(f"{axis_name} axis — Stability")
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


    if consolidated_frequency_regions is None:
        raise RuntimeError("Consolidated frequency regions were not built")

    trusted_fig, trusted_axes = plt.subplots(
        3,
        1,
        figsize=(14, 10),
        sharex=True,
        constrained_layout=True,
    )

    trusted_regions_by_axis = {
        "X": consolidated_frequency_regions.x,
        "Y": consolidated_frequency_regions.y,
        "Z": consolidated_frequency_regions.z,
    }

    for trusted_axis, (axis_name, axis_data) in zip(
        trusted_axes,
        visualization_axes.items(),
    ):
        trusted_axis.plot(
            axis_data.frequency,
            axis_data.trusted_frequency_psd,
            color=COLORS[axis_name],
            label=f"{axis_name} Trusted Median PSD",
        )
        for region in trusted_regions_by_axis[axis_name]:
            if not is_trusted_frequency_cluster(
                region,
                config.visualization.trusted_frequency,
            ):
                continue
            evidence = region.median_evidence
            matching_indices = np.flatnonzero(
                axis_data.frequency == evidence.peak_frequency
            )
            if len(matching_indices) != 1:
                raise ValueError(
                    f"Trusted Median peak frequency {evidence.peak_frequency} Hz "
                    f"does not match exactly one {axis_name} frequency bin"
                )
            peak_psd = axis_data.trusted_frequency_psd[matching_indices[0]]
            trusted_axis.scatter(
                evidence.peak_frequency,
                peak_psd,
                color=COLORS[axis_name],
                marker="x",
            )
            trusted_axis.annotate(
                f"{evidence.peak_frequency:.1f} Hz\n"
                f"{region.support_count}/"
                f"{aligned_psd.x_stack.shape[0]}\n"
                f"{evidence.prominence_db:.2f} dB",
                xy=(evidence.peak_frequency, peak_psd),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        trusted_axis.set_title(f"{axis_name} axis — Trusted Median PSD")
        trusted_axis.set_ylabel("Trusted PSD [g²/Hz]")
        trusted_axis.set_ylim(bottom=0)
        trusted_axis.set_xlim(
            analysis_min_frequency,
            analysis_max_frequency,
        )
        trusted_axis.grid(True, alpha=0.25, linewidth=0.6)
        if axis_name == "X":
            trusted_axis.legend()

    trusted_axes[-1].set_xlabel("Frequency, Hz")
    trusted_fig.suptitle("Trusted Median PSD")

    if run_result_paths is None:
        raise RuntimeError("Run result paths were not built")

    save_run_figures(run_result_paths, stat_fig, trusted_fig)
    print("Saved results:")
    print(f"  {run_result_paths.log}")
    print(f"  {run_result_paths.figure1}")
    print(f"  {run_result_paths.figure2}")
    if run_result_paths.raw.exists():
        print(f"  {run_result_paths.raw}")

    if show_figures:
        plt.show()
    else:
        plt.close(stat_fig)
        plt.close(trusted_fig)

    return stop_requested


try:
    for run_number in range(1, total_runs + 1):
        stop_series = run_measurement(
            run_number,
            total_runs,
            show_figures=(total_runs == 1),
        )
        if stop_series:
            break
finally:
    ser.close()
