import argparse
import csv
import re
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
REPLAY_RESULTS_DIRECTORY = Path("replay_results")

# AXIS = "X"          # X / Y / Z

MAGIC = b"VIB2"

SUPPORTED_ODR_HZ = {250.0, 125.0, 62.5}

SET_ODR_COMMAND = 0x01
ODR_PARAMETER_BY_HZ = {
    250.0: 0x00,
    125.0: 0x01,
    62.5: 0x02,
}

@dataclass(frozen=True)
class Theme:
    """Colours and line weights of the figures and of the viewer window.

    THEMES holds four: light and dark, each with a high-contrast variant.
    The series colours of X, Y and Z were checked with a colour-vision
    validator on their surface; the high-contrast variants raise contrast
    moderately, they do not go to pure black on white.
    """

    name: str
    figure_background: str
    axes_background: str
    text: str
    muted_text: str
    grid: str
    spine: str
    series: dict[str, str]
    # Band boundaries and threshold lines of figure 1.
    accent: str
    # The local window power stability curve of figure 1.
    secondary: str
    tooltip_background: str
    tooltip_border: str
    line_width: float


THEMES: dict[str, Theme] = {
    "light": Theme(
        name="light",
        figure_background="#f4f4f1",
        axes_background="#ffffff",
        text="#1f2328",
        muted_text="#5b6168",
        grid="#dcdee2",
        spine="#b8bcc2",
        series={"X": "#2a78d6", "Y": "#eb6834", "Z": "#1baf7a"},
        accent="#6b7280",
        secondary="#8b5cf6",
        tooltip_background="#ffffff",
        tooltip_border="#9aa0a6",
        line_width=1.4,
    ),
    "light-contrast": Theme(
        name="light-contrast",
        figure_background="#ffffff",
        axes_background="#ffffff",
        text="#000000",
        muted_text="#2d3136",
        grid="#b4b8be",
        spine="#000000",
        series={"X": "#1c5cab", "Y": "#c2491c", "Z": "#0f7a52"},
        accent="#3b3f45",
        secondary="#6d28d9",
        tooltip_background="#ffffff",
        tooltip_border="#000000",
        line_width=1.8,
    ),
    "dark": Theme(
        name="dark",
        figure_background="#1b1d21",
        axes_background="#24262b",
        text="#e3e6ea",
        muted_text="#9aa1aa",
        grid="#363a41",
        spine="#4a4f58",
        series={"X": "#3987e5", "Y": "#d95926", "Z": "#199e70"},
        accent="#9aa1aa",
        secondary="#b39ddb",
        tooltip_background="#2c2f36",
        tooltip_border="#5b616b",
        line_width=1.4,
    ),
    "dark-contrast": Theme(
        name="dark-contrast",
        figure_background="#0e0f11",
        axes_background="#121315",
        text="#fafafa",
        muted_text="#cfd3d8",
        grid="#4a5059",
        spine="#8a9099",
        series={"X": "#4f93e8", "Y": "#e5602f", "Z": "#1fa872"},
        accent="#cfd3d8",
        secondary="#c4b5fd",
        tooltip_background="#1c1e22",
        tooltip_border="#c0c4ca",
        line_width=1.8,
    ),
}
DEFAULT_THEME = "light"


def resolve_theme(name: str) -> Theme:
    try:
        return THEMES[name]
    except KeyError:
        raise ValueError(
            f"unknown theme {name!r}; one of {', '.join(THEMES)}"
        ) from None


def theme_rc_params(theme: Theme) -> dict[str, Any]:
    """matplotlib settings of a theme, for ``plt.rc_context`` around drawing.

    Only artists created inside the context take them, so a redraw of the
    viewer window recreates its axes under the chosen theme.
    """
    return {
        "figure.facecolor": theme.figure_background,
        "figure.edgecolor": theme.figure_background,
        "figure.titlesize": 13,
        "figure.titleweight": "bold",
        "axes.facecolor": theme.axes_background,
        "axes.edgecolor": theme.spine,
        "axes.linewidth": 0.8,
        "axes.labelcolor": theme.text,
        "axes.labelsize": 10,
        "axes.titlecolor": theme.text,
        "axes.titlelocation": "left",
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlepad": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": theme.spine,
        "ytick.color": theme.spine,
        "xtick.labelcolor": theme.muted_text,
        "ytick.labelcolor": theme.muted_text,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "grid.color": theme.grid,
        "grid.alpha": 1.0,
        "grid.linewidth": 0.6,
        "text.color": theme.text,
        "legend.frameon": False,
        "legend.labelcolor": theme.text,
        "legend.fontsize": 9,
        "lines.linewidth": theme.line_width,
    }

# Session layouts, packets per session x sessions per run, are listed per
# old detector mode in config.toml with their Med.Prom thresholds; replay and
# live runs use only those.
LAYOUT_PATTERN = r"[1-9][0-9]*x[1-9][0-9]*"

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
    # The threshold of the run's layout in the active old detector mode;
    # None until resolve_layout_trusted_threshold sets it for a run.
    min_median_prominence_db: float | None
    background_weight: float
    min_band_contrast_db: float
    weak_trusted_weight: float


@dataclass(frozen=True)
class OldDetectorModeConfig:
    """One old detector mode, [old_detector.nperseg_<n>] in config.toml.

    ``thresholds`` holds (layout, Med.Prom dB) in config order: the layouts
    this mode is calibrated for. A shorter run has a noisier median PSD and
    needs a higher Med.Prom; nperseg 2048 was calibrated on synthetic noise
    on 2026-10-03, nperseg 1024 keeps its single 1.55 dB.
    """

    nperseg: int
    noverlap: int
    thresholds: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class FrequencyClusterConsolidationConfig:
    median_frequency_tolerance_hz: float
    # How many sessions two clusters may share and still merge. 0 is the
    # rule until 2026-10-04: their sessions must not overlap at all.
    max_shared_sessions: int = 0


@dataclass(frozen=True)
class FrequencyClusteringConfig:
    frequency_tolerance_hz_250: float
    frequency_tolerance_hz_125: float
    frequency_tolerance_hz_62p5: float


@dataclass(frozen=True)
class VisualizationConfig:
    trusted_frequency: TrustedFrequencyVisualizationConfig
    # A key of THEMES: [visualization] theme in config.toml, --theme for a run.
    theme: str = DEFAULT_THEME


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
    old_detector_modes: tuple[OldDetectorModeConfig, ...] = ()


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
    if trusted_frequency.min_median_prominence_db is not None:
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
    if (
        trusted_frequency.min_median_prominence_db is not None
        and trusted_frequency.min_median_prominence_db < 0
    ):
        raise ValueError(
            "Visualization trusted frequency min_median_prominence_db must "
            "be non-negative"
        )
    if not config.old_detector_modes:
        raise ValueError("config.toml needs an [old_detector.nperseg_<n>] section")
    for mode in config.old_detector_modes:
        name = f"[old_detector.nperseg_{mode.nperseg}]"
        require_positive_integer(mode.nperseg, f"{name} nperseg")
        require_non_negative_integer(mode.noverlap, f"{name} noverlap")
        if mode.noverlap >= mode.nperseg:
            raise ValueError(f"{name} noverlap must be less than nperseg")
        if not mode.thresholds:
            raise ValueError(
                f"{name} needs min_median_prominence_db_by_layout with at "
                "least one layout"
            )
        for layout, value in mode.thresholds:
            if not re.fullmatch(LAYOUT_PATTERN, layout):
                raise ValueError(
                    f"{name} layouts must look like 8x16, not {layout!r}"
                )
            require_finite_number(value, f"{name} Med.Prom threshold of {layout}")
            if value < 0:
                raise ValueError(
                    f"{name} Med.Prom threshold of {layout} must be non-negative"
                )
    active_old_detector_mode(config)
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
    theme_name = str(raw_config["visualization"].get("theme", DEFAULT_THEME))
    resolve_theme(theme_name)
    consolidation_data = raw_config["analysis"][
        "frequency_cluster_consolidation"
    ]
    frequency_clustering_data = raw_config["analysis"][
        "frequency_clustering"
    ]
    band_entries = raw_config["analysis"]["bands"]

    sensor_odr_hz = float(sensor_data["odr_hz"])
    old_detector_modes = tuple(
        OldDetectorModeConfig(
            nperseg=int(name.removeprefix("nperseg_")),
            noverlap=mode_data["noverlap"],
            thresholds=tuple(
                (str(layout), float(value))
                for layout, value in mode_data[
                    "min_median_prominence_db_by_layout"
                ].items()
            ),
        )
        for name, mode_data in raw_config.get("old_detector", {}).items()
        if re.fullmatch(r"nperseg_[1-9][0-9]*", name)
    )
    welch_nperseg = welch_data["nperseg"]
    welch_mode = next(
        (mode for mode in old_detector_modes if mode.nperseg == welch_nperseg),
        None,
    )
    if welch_mode is None:
        raise ValueError(
            f"[welch] nperseg {welch_nperseg} has no "
            f"[old_detector.nperseg_{welch_nperseg}] section"
        )
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
            nperseg=welch_mode.nperseg,
            noverlap=welch_mode.noverlap,
        ),
        visualization=VisualizationConfig(
            trusted_frequency=TrustedFrequencyVisualizationConfig(
                min_support_fraction=trusted_frequency_data[
                    "min_support_fraction"
                ],
                min_median_prominence_db=None,
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
            theme=theme_name,
        ),
        frequency_clustering=frequency_clustering,
        frequency_cluster_consolidation=FrequencyClusterConsolidationConfig(
            median_frequency_tolerance_hz=consolidation_data[
                "median_frequency_tolerance_hz"
            ],
            max_shared_sessions=consolidation_data.get(
                "max_shared_sessions", 0,
            ),
        ),
        analysis_bands=analysis_bands,
        old_detector_modes=old_detector_modes,
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
    parser.add_argument("--replay", type=Path)
    parser.add_argument(
        "--virtual-mode",
        nargs="+",
        default=["all"],
        help="replay layouts, packets per session x sessions per run, e.g. "
             "8x16; only those listed for the mode in config.toml; all (the "
             "default) = every layout of the mode",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="with --replay and one --virtual-mode layout: open figure 2 of "
             "its virtual runs in one interactive window instead of writing "
             "files",
    )
    parser.add_argument(
        "--nperseg",
        type=int,
        default=None,
        help="old detector mode for this run, e.g. 1024 or 2048; it needs an "
             "[old_detector.nperseg_<n>] section (default: [welch] nperseg)",
    )
    parser.add_argument(
        "--theme",
        choices=list(THEMES),
        default=None,
        help="colours of the figures and the viewer window "
             "(default: [visualization] theme, light)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="configuration file (default: config.toml next to main.py)",
    )
    parser.add_argument(
        "--replay-root",
        type=Path,
        default=REPLAY_RESULTS_DIRECTORY,
        help="where --replay writes its <record> folder "
             "(default: replay_results)",
    )
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
    nperseg = getattr(arguments, "nperseg", None)
    if nperseg is not None:
        config = select_old_detector_mode(config, nperseg)
    theme_name = getattr(arguments, "theme", None)
    if theme_name is not None:
        resolve_theme(theme_name)
        config = replace(
            config,
            visualization=replace(config.visualization, theme=theme_name),
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


def parabola_peak(
    frequency: np.ndarray,
    psd: np.ndarray,
    index: int,
) -> tuple[float, float]:
    """Top of a peak between bins, as stable_spectrum.py finds it.

    A parabola through the log PSD of the peak bin and its two neighbours
    puts the top of a Hann-windowed peak to a small fraction of a bin; the
    shift is kept within half a bin. Returns the top's frequency and PSD. A
    bin at the edge or one that is not above both neighbours stays as it is.
    Only what is reported uses this; the detector decides on bin centres.
    """
    bin_frequency = float(frequency[index])
    bin_psd = float(psd[index])
    if index <= 0 or index >= len(psd) - 1:
        return bin_frequency, bin_psd
    tiny = np.finfo(float).tiny
    left, centre, right = 10.0 * np.log10(
        np.maximum(np.asarray(psd[index - 1:index + 2], dtype=float), tiny)
    )
    curvature = left - 2.0 * centre + right
    if not curvature < 0.0:
        return bin_frequency, bin_psd
    step = float(frequency[index + 1] - frequency[index])
    offset = float(np.clip(0.5 * (left - right) / curvature, -0.5, 0.5))
    top_db = centre + 0.5 * (right - left) * offset + 0.5 * curvature * offset ** 2
    return bin_frequency + offset * step, float(10.0 ** (top_db / 10.0))


def peak_tops_on_curve(
    frequency: np.ndarray,
    psd: np.ndarray,
    peak_frequencies,
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]]]:
    """The curve to draw, with the top of each reported peak as a vertex.

    Each peak is given by the centre of its bin. Its parabola top is added
    to the plotted points, so the line runs through it and the cross put
    there sits on the curve at the reported frequency. Only the figure gets
    the extra points.
    """
    tops = []
    for peak_frequency in peak_frequencies:
        matching_indices = np.flatnonzero(frequency == peak_frequency)
        if len(matching_indices) != 1:
            raise ValueError(
                f"Peak frequency {peak_frequency} Hz does not match exactly one "
                "frequency bin"
            )
        tops.append(parabola_peak(frequency, psd, int(matching_indices[0])))
    extra = [top for top in tops if not np.any(frequency == top[0])]
    curve_frequency = np.concatenate([frequency, [top[0] for top in extra]])
    curve_psd = np.concatenate([psd, [top[1] for top in extra]])
    order = np.argsort(curve_frequency, kind="stable")
    return curve_frequency[order], curve_psd[order], tops


def session_layout_name(config: ApplicationConfig) -> str:
    return (
        f"{config.session.packets_per_session}"
        f"x{config.session.min_recommended_sessions}"
    )


def parse_layout(layout: str) -> tuple[int, int]:
    """``8x16`` -> 8 packets per session, 16 sessions per run."""
    if not re.fullmatch(LAYOUT_PATTERN, layout):
        raise ValueError(f"layout {layout!r} must look like 8x16")
    packets_per_session, sessions = layout.split("x")
    return int(packets_per_session), int(sessions)


def active_old_detector_mode(config: ApplicationConfig) -> OldDetectorModeConfig:
    mode = next(
        (
            mode for mode in config.old_detector_modes
            if mode.nperseg == config.welch.nperseg
        ),
        None,
    )
    if mode is None:
        raise ValueError(
            f"Welch nperseg {config.welch.nperseg} has no "
            f"[old_detector.nperseg_{config.welch.nperseg}] section; "
            f"modes: {old_detector_mode_names(config)}"
        )
    return mode


def old_detector_mode_names(config: ApplicationConfig) -> str:
    return ", ".join(str(mode.nperseg) for mode in config.old_detector_modes)


def mode_layouts(config: ApplicationConfig) -> list[str]:
    """Layouts the active mode is calibrated for, in config order."""
    return [layout for layout, _ in active_old_detector_mode(config).thresholds]


def select_old_detector_mode(
    config: ApplicationConfig,
    nperseg: int,
) -> ApplicationConfig:
    mode = next(
        (mode for mode in config.old_detector_modes if mode.nperseg == nperseg),
        None,
    )
    if mode is None:
        raise ValueError(
            f"--nperseg {nperseg} has no [old_detector.nperseg_{nperseg}] "
            f"section in config.toml; modes: {old_detector_mode_names(config)}"
        )
    return replace(
        config,
        welch=replace(config.welch, nperseg=mode.nperseg, noverlap=mode.noverlap),
    )


def resolve_layout_trusted_threshold(
    config: ApplicationConfig,
) -> ApplicationConfig:
    """The run config with the Med.Prom threshold of its own layout.

    The layout must be listed for the active mode: an uncalibrated layout
    is refused rather than judged with another layout's threshold.
    """
    layout = session_layout_name(config)
    thresholds = dict(active_old_detector_mode(config).thresholds)
    if layout not in thresholds:
        raise ValueError(
            f"layout {layout} has no Med.Prom threshold for nperseg "
            f"{config.welch.nperseg}; calibrated layouts: "
            f"{', '.join(thresholds)}"
        )
    trusted_frequency = config.visualization.trusted_frequency
    return replace(
        config,
        visualization=replace(
            config.visualization,
            trusted_frequency=replace(
                trusted_frequency,
                min_median_prominence_db=thresholds[layout],
            ),
        ),
    )


def trusted_threshold_line(config: ApplicationConfig) -> str:
    threshold = config.visualization.trusted_frequency.min_median_prominence_db
    return (
        f"Trusted Med.Prom threshold: {threshold:g} dB "
        f"(layout {session_layout_name(config)}, "
        f"nperseg {config.welch.nperseg})\n"
        "Clusters merge with up to "
        f"{config.frequency_cluster_consolidation.max_shared_sessions} "
        "shared session(s)\n"
    )


def resolve_replay_layouts(
    virtual_modes: list[str],
    config: ApplicationConfig,
) -> list[str]:
    available = mode_layouts(config)
    layouts = []
    for mode in virtual_modes:
        if mode == "all":
            layouts.extend(available)
        elif mode in available:
            layouts.append(mode)
        else:
            raise ValueError(
                f"replay layout {mode} has no Med.Prom threshold for nperseg "
                f"{config.welch.nperseg}; calibrated layouts: "
                f"{', '.join(available)}"
            )
    return list(dict.fromkeys(layouts))


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
            + trusted_threshold_line(config)
            + "\n"
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
    # Top of the parabola through the candidate bin and its neighbours,
    # reported instead of the bin centre; decisions use ``frequency``.
    refined_frequency: float | None = None

    @property
    def reported_frequency(self) -> float:
        if self.refined_frequency is not None:
            return self.refined_frequency
        return self.frequency


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
    # Centre of the peak bin: every decision of the detector uses it.
    peak_frequency: float | None
    peak_psd: float | None
    prominence_db: float | None
    local_contrast_db: float | None
    band_contrast_db: float | None
    passed_prominence: bool | None
    # Top of the parabola through the peak bin and its neighbours: what the
    # tables, CSV files and figures report as Med.Freq since 2026-10-03.
    refined_frequency: float | None = None
    # How far the parabola top rises above the peak bin, in dB.
    refined_rise_db: float | None = None


def reported_median_frequency(evidence: MedianPSDEvidence) -> float | None:
    if evidence.refined_frequency is not None:
        return evidence.refined_frequency
    return evidence.peak_frequency


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
    shared = config.max_shared_sessions
    if not isinstance(shared, int) or isinstance(shared, bool) or shared < 0:
        raise ValueError(
            "Frequency cluster consolidation max_shared_sessions must be a "
            "non-negative integer"
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
            # Sessions with a peak in both clusters: two lines side by side,
            # or one line whose session also caught a noise peak nearby.
            if (
                len(diagnostic_sessions & group_sessions)
                > config.max_shared_sessions
            ):
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


@dataclass(frozen=True)
class AnalysisResult:
    aligned_psd: AlignedPSDData
    statistics: StatisticsResult
    candidate_diagnostics: PeakCandidateDiagnostics
    peaks: PeakResult
    frequency_clusters: FrequencyClusterResult
    frequency_cluster_diagnostics: FrequencyClusterDiagnostics
    consolidated_frequency_regions: ConsolidatedFrequencyRegions
    visualization_data: VisualizationData


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
                f"{item.band_name:<{band_width}}  {item.reported_frequency:7.2f}  "
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
    refined_frequency, refined_psd = parabola_peak(
        frequency, median_psd, global_peak_index,
    )
    return MedianPSDEvidence(
        peak_frequency=peak_frequency,
        refined_frequency=refined_frequency,
        refined_rise_db=float(
            10.0 * np.log10(max(refined_psd, tiny) / safe_peak_psd)
        ),
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
                f"{reported_median_frequency(evidence):.2f}"
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
                f"{reported_median_frequency(evidence):.2f}"
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
                f"{reported_median_frequency(evidence):8.2f}  "
                f"{evidence.prominence_db:8.2f}  "
                f"{median_contrast:>9}  "
                f"{band_contrast:>10}  "
                f"{weight:6.2f}  "
                f"{region.minimum_frequency:.2f}–"
                f"{region.maximum_frequency:.2f}"
            )


def select_candidate_regions(
    axis_regions: list[ConsolidatedFrequencyRegion],
    config: TrustedFrequencyVisualizationConfig,
) -> list[ConsolidatedFrequencyRegion]:
    """The one region below the Med.Prom threshold worth a look, on an axis
    with no trusted region; for the user to judge, never a result.

    Eligible are regions with enough support whose median PSD maximum lies
    inside their own range of session peaks: a region given the maximum of a
    neighbour through the search window says nothing about its own frequency.
    The candidate is the highest Med.Prom, then the larger support. Decisions
    of the detector do not depend on it.
    """
    if any(is_trusted_frequency_cluster(region, config) for region in axis_regions):
        return []
    eligible = [
        region for region in axis_regions
        if region.support_fraction >= config.min_support_fraction
        and region.median_evidence.peak_frequency is not None
        and region.median_evidence.prominence_db is not None
        and region.minimum_frequency
        <= region.median_evidence.peak_frequency
        <= region.maximum_frequency
    ]
    if not eligible:
        return []
    return [max(
        eligible,
        key=lambda region: (
            region.median_evidence.prominence_db, region.support_count,
        ),
    )]


def print_candidate_regions(
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
        print(f"Candidates below the threshold — {axis_name}")
        if any(is_trusted_frequency_cluster(region, config) for region in axis_regions):
            print("Trusted frequency regions present; no candidates.")
            continue
        candidates = select_candidate_regions(axis_regions, config)
        if not candidates:
            print("No candidate regions.")
            continue
        print(
            f"Below Med.Prom {config.min_median_prominence_db:g} dB: not a "
            "result, noise reaches such levels too; open circles on figure 2."
        )
        print(
            f"{'Freq Hz':>7}  {'Support':>7}  "
            f"{'Med.Freq':>8}  {'Med.Prom':>8}  {'Range Hz':>13}"
        )
        for region in candidates:
            evidence = region.median_evidence
            print(
                f"{region.frequency:7.2f}  "
                f"{region.support_count:>3}/{total_sessions:<3}  "
                f"{reported_median_frequency(evidence):8.2f}  "
                f"{evidence.prominence_db:8.2f}  "
                f"{region.minimum_frequency:.2f}–"
                f"{region.maximum_frequency:.2f}"
            )


def draw_analysis_bands(
    ax,
    analysis_bands: list[AnalysisBand],
    theme: Theme | None = None,
) -> None:
    if theme is None:
        theme = THEMES[DEFAULT_THEME]
    boundaries = sorted({
        boundary
        for band in analysis_bands
        for boundary in (band.min_frequency, band.max_frequency)
    })

    for boundary in boundaries:
        ax.axvline(
            boundary,
            color=theme.accent,
            linestyle="--",
            linewidth=0.8,
            alpha=0.6,
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
            color=theme.muted_text,
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
            f"{frequency:.2f} Hz\n"
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
                            refined_frequency=parabola_peak(
                                freq, median, int(global_peak_index),
                            )[0],
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
                    refined_frequency=parabola_peak(
                        freq, median, int(global_peak_index),
                    )[0],
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
    CONFIG_PATH = cli_arguments.config
    REPLAY_RESULTS_DIRECTORY = cli_arguments.replay_root
    config = build_effective_config_from_cli(CONFIG_PATH, cli_arguments)
    if cli_arguments.show and (
        cli_arguments.replay is None
        or len(cli_arguments.virtual_mode) != 1
        or cli_arguments.virtual_mode[0] == "all"
    ):
        raise ValueError(
            "--show needs --replay and --virtual-mode with one layout, "
            "e.g. --virtual-mode 8x16"
        )
    if cli_arguments.replay is None:
        # A live run needs a threshold for its own layout, checked before the
        # sensor is opened; replay resolves the threshold of each layout.
        config = resolve_layout_trusted_threshold(config)
    else:
        resolve_replay_layouts(cli_arguments.virtual_mode, config)
except FileNotFoundError:
    raise SystemExit(f"Configuration file not found: {CONFIG_PATH}")
except tomllib.TOMLDecodeError as error:
    raise SystemExit(f"Invalid TOML configuration: {error}")
except (KeyError, TypeError, ValueError) as error:
    raise SystemExit(f"Invalid configuration: {error}")

total_runs = cli_arguments.repeat

analysis_bands = config.analysis_bands
analysis_min_frequency, analysis_max_frequency = (
    get_analysis_frequency_limits(analysis_bands)
)

ser = None
if cli_arguments.replay is None:
    print(f"ODR: {config.sensor.odr_hz:g} Hz")
    print(f"Packets/session: {config.session.packets_per_session}")
    print(f"Target sessions: {config.session.min_recommended_sessions}")
    print(
        "Frequency tolerance: "
        f"{resolve_frequency_tolerance_hz(config.frequency_clustering, config.sensor.odr_hz):.2f} "
        "Hz"
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

def compute_average_fft(signal, fs, welch_config):

    nperseg = min(welch_config.nperseg, len(signal))
    noverlap = min(welch_config.noverlap, nperseg // 2)
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

def compute_psd(signal, fs, welch_config, frequency_limits):

    freq, psd = welch(
        signal,
        fs=fs,
        window="hann",
        nperseg=min(welch_config.nperseg, len(signal)),
        noverlap=min(welch_config.noverlap, len(signal)//2),
        scaling="density"
    )

    mask = (
        (freq >= frequency_limits[0])
        & (freq <= frequency_limits[1])
    )

    freq = freq[mask]
    psd = psd[mask]

    resolution = fs / min(welch_config.nperseg, len(signal))

    return PSDResult(
        freq=freq,
        psd=psd,
        resolution=resolution
    )

def process_session(session, session_fs, number, run_config):

    if any(
        len(session[axis]) != run_config.session.packets_per_session
        for axis in ("X", "Y", "Z")
    ):
        raise ValueError("Cannot process incomplete session")

    if len(session_fs) != run_config.session.packets_per_session:
        raise ValueError("Cannot process incomplete session")

    fs = np.mean(session_fs)

    axis_results = {}
    frequency_limits = get_analysis_frequency_limits(
        run_config.analysis_bands
    )

    for axis in ("X", "Y", "Z"):

        signal = np.concatenate(session[axis])

        signal = signal - np.mean(signal)

        axis_results[axis] = AxisResult(
            fft=compute_fft(signal, fs),
            average_fft=compute_average_fft(signal, fs, run_config.welch),
            psd=compute_psd(
                signal,
                fs,
                run_config.welch,
                frequency_limits,
            )
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


def build_replay_config(
    base_config: ApplicationConfig,
    raw: RawMeasurement,
    packets_per_session: int,
    target_sessions: int,
) -> ApplicationConfig:
    frequency_tolerance_hz = resolve_frequency_tolerance_hz(
        base_config.frequency_clustering,
        raw.requested_odr_hz,
    )
    replay_config = replace(
        base_config,
        sensor=replace(
            base_config.sensor,
            odr_hz=raw.requested_odr_hz,
        ),
        session=replace(
            base_config.session,
            packets_per_session=packets_per_session,
            min_recommended_sessions=target_sessions,
        ),
        analysis_bands=[
            replace(
                band,
                frequency_tolerance_hz=frequency_tolerance_hz,
            )
            for band in base_config.analysis_bands
        ],
    )
    validate_config(replay_config)
    return resolve_layout_trusted_threshold(replay_config)


def build_virtual_run_slices(
    packet_count: int,
    packets_per_session: int,
    target_sessions: int,
) -> list[tuple[int, int]]:
    packets_per_run = packets_per_session * target_sessions
    if packets_per_run <= 0:
        raise ValueError("Virtual layout dimensions must be positive")
    run_count = packet_count // packets_per_run
    return [
        (
            run_index * packets_per_run,
            (run_index + 1) * packets_per_run,
        )
        for run_index in range(run_count)
    ]


def build_sessions_from_raw(
    raw: RawMeasurement,
    start_packet: int,
    end_packet: int,
    run_config: ApplicationConfig,
) -> list[SessionResult]:
    validate_raw_measurement(raw)
    if not 0 <= start_packet < end_packet <= raw.x.shape[0]:
        raise ValueError("Virtual packet range is outside raw measurement")
    packet_count = end_packet - start_packet
    packets_per_session = run_config.session.packets_per_session
    if packet_count % packets_per_session != 0:
        raise ValueError("Virtual packet range does not contain full sessions")
    sessions = []
    for session_offset in range(0, packet_count, packets_per_session):
        packet_slice = slice(
            start_packet + session_offset,
            start_packet + session_offset + packets_per_session,
        )
        session = {
            "X": list(raw.x[packet_slice]),
            "Y": list(raw.y[packet_slice]),
            "Z": list(raw.z[packet_slice]),
        }
        sessions.append(
            process_session(
                session,
                list(raw.packet_fs_hz[packet_slice]),
                len(sessions) + 1,
                run_config,
            )
        )
    if len(sessions) != run_config.session.min_recommended_sessions:
        raise ValueError("Virtual packet range produced unexpected session count")
    return sessions


def analyze_sessions(
    sessions: list[SessionResult],
    run_config: ApplicationConfig,
) -> AnalysisResult:
    aligned_psd = build_aligned_psd_data(sessions)
    statistics = compute_statistics(aligned_psd)
    candidate_diagnostics = PeakCandidateDiagnostics(x=[], y=[], z=[])
    peaks = find_psd_peaks(
        statistics,
        aligned_psd,
        run_config.analysis_bands,
        candidate_diagnostics,
    )
    frequency_clusters = build_session_frequency_clusters(
        aligned_psd,
        run_config.analysis_bands,
    )
    frequency_cluster_diagnostics = (
        build_session_frequency_cluster_diagnostics(
            statistics,
            frequency_clusters,
            run_config.analysis_bands,
            aligned_psd.frequency,
        )
    )
    consolidated_frequency_regions = build_consolidated_frequency_regions(
        frequency_cluster_diagnostics,
        run_config.analysis_bands,
        aligned_psd.x_stack.shape[0],
        run_config.frequency_cluster_consolidation,
    )
    visualization_data = build_visualization_data(
        statistics,
        peaks,
        aligned_psd,
        run_config.analysis_bands,
        consolidated_frequency_regions,
        run_config.visualization.trusted_frequency,
    )
    return AnalysisResult(
        aligned_psd=aligned_psd,
        statistics=statistics,
        candidate_diagnostics=candidate_diagnostics,
        peaks=peaks,
        frequency_clusters=frequency_clusters,
        frequency_cluster_diagnostics=frequency_cluster_diagnostics,
        consolidated_frequency_regions=consolidated_frequency_regions,
        visualization_data=visualization_data,
    )


# ==========================================================

AXIS_NAMES = ("X", "Y", "Z")


@dataclass
class HoverPanel:
    """One axis panel of figure 2 and what the mouse can point at in it."""

    ax: Any
    axis_name: str
    frequency: np.ndarray
    psd: np.ndarray
    # (frequency, PSD, text) of each marked peak: trusted or candidate.
    peaks: list[tuple[float, float, str]]
    annotation: Any


def peak_hover_text(
    axis_name: str,
    region: ConsolidatedFrequencyRegion,
    total_sessions: int,
    config: TrustedFrequencyVisualizationConfig,
    candidate: bool = False,
) -> str:
    evidence = region.median_evidence
    kind = (
        "candidate below the threshold" if candidate else "trusted"
    )
    lines = [
        f"{axis_name} {reported_median_frequency(evidence):.2f} Hz — {kind}",
        f"Med.Freq bin {evidence.peak_frequency:.3f} Hz",
        f"Freq {region.frequency:.2f} Hz, session peaks "
        f"{region.minimum_frequency:.2f}–{region.maximum_frequency:.2f}",
        f"support {region.support_count}/{total_sessions}",
        f"Med.Prom {evidence.prominence_db:.2f} dB "
        f"(threshold {config.min_median_prominence_db:g})",
    ]
    if evidence.band_contrast_db is not None:
        lines.append(
            f"Band.Contr {evidence.band_contrast_db:.2f} dB, weight "
            f"{get_trusted_frequency_weight(region, config):.1f}"
        )
    return "\n".join(lines)


def draw_trusted_figure(
    fig,
    analysis_result: AnalysisResult,
    run_config: ApplicationConfig,
    axis_names: tuple[str, ...] = AXIS_NAMES,
    title: str = "Trusted Median PSD",
    theme: Theme | None = None,
) -> list[HoverPanel]:
    """Figure 2 on ``fig``: one panel per axis in ``axis_names``.

    Returns the panels with what hover tooltips need. The saved figure 2
    and the interactive window draw it alike, in ``theme`` or in the one of
    ``run_config``.
    """
    if theme is None:
        theme = resolve_theme(run_config.visualization.theme)
    with plt.rc_context(theme_rc_params(theme)):
        return draw_trusted_panels(
            fig, analysis_result, run_config, axis_names, title, theme,
        )


def draw_trusted_panels(
    fig,
    analysis_result: AnalysisResult,
    run_config: ApplicationConfig,
    axis_names: tuple[str, ...],
    title: str,
    theme: Theme,
) -> list[HoverPanel]:
    fig.clear()
    fig.set_facecolor(theme.figure_background)
    fig.suptitle(title)
    if not axis_names:
        fig.text(0.5, 0.5, "All axes are hidden", ha="center", va="center")
        return []
    aligned_psd = analysis_result.aligned_psd
    visualization_data = analysis_result.visualization_data
    regions = analysis_result.consolidated_frequency_regions
    trusted_config = run_config.visualization.trusted_frequency
    total_sessions = aligned_psd.x_stack.shape[0]
    analysis_min_frequency, analysis_max_frequency = (
        get_analysis_frequency_limits(run_config.analysis_bands)
    )
    visualization_axes = {
        "X": visualization_data.x,
        "Y": visualization_data.y,
        "Z": visualization_data.z,
    }
    regions_by_axis = {"X": regions.x, "Y": regions.y, "Z": regions.z}
    axes = fig.subplots(len(axis_names), 1, sharex=True, squeeze=False)[:, 0]
    panels = []
    for panel_index, (trusted_axis, axis_name) in enumerate(zip(axes, axis_names)):
        axis_data = visualization_axes[axis_name]
        trusted_regions = [
            region for region in regions_by_axis[axis_name]
            if is_trusted_frequency_cluster(region, trusted_config)
        ]
        candidates = select_candidate_regions(
            regions_by_axis[axis_name], trusted_config,
        )
        # The tops of the plotted curve: Med.Freq is computed on the median
        # PSD, and the trusted weight is the same over the peak.
        curve_frequency, curve_psd, tops = peak_tops_on_curve(
            axis_data.frequency,
            axis_data.trusted_frequency_psd,
            [
                region.median_evidence.peak_frequency
                for region in (
                    *trusted_regions, *candidates,
                )
            ],
        )
        trusted_axis.plot(
            curve_frequency,
            curve_psd,
            color=theme.series[axis_name],
            label=f"{axis_name} Trusted Median PSD",
        )
        hover_peaks = []
        for region, (peak_frequency, peak_psd) in zip(
            trusted_regions, tops[:len(trusted_regions)],
        ):
            trusted_axis.scatter(
                peak_frequency, peak_psd, color=theme.series[axis_name],
                marker="x", s=50, linewidths=1.6, zorder=4,
            )
            trusted_axis.annotate(
                f"{reported_median_frequency(region.median_evidence):.2f} Hz\n"
                f"{region.support_count}/{total_sessions}",
                xy=(peak_frequency, peak_psd),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color=theme.text,
            )
            hover_peaks.append((
                peak_frequency, peak_psd,
                peak_hover_text(axis_name, region, total_sessions, trusted_config),
            ))
        for region, (top_frequency, top_psd) in zip(
            candidates, tops[len(trusted_regions):],
        ):
            # Open circle: the strongest of what stayed below the threshold.
            trusted_axis.scatter(
                top_frequency,
                top_psd,
                facecolors="none",
                edgecolors=theme.series[axis_name],
                marker="o",
                s=45,
                linewidths=1.2,
                zorder=4,
            )
            trusted_axis.annotate(
                f"{reported_median_frequency(region.median_evidence):.2f} Hz\n"
                f"{region.support_count}/{total_sessions}",
                xy=(top_frequency, top_psd),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                color=theme.muted_text,
            )
            hover_peaks.append((
                top_frequency, top_psd,
                peak_hover_text(
                    axis_name, region, total_sessions, trusted_config, True,
                ),
            ))
        trusted_axis.set_title(f"{axis_name} axis — Trusted Median PSD")
        trusted_axis.set_ylabel("Trusted PSD [g²/Hz]")
        # Headroom so the labels of the highest peaks stay under the title.
        trusted_axis.set_ylim(0, float(np.max(curve_psd)) * 1.22)
        trusted_axis.set_xlim(analysis_min_frequency, analysis_max_frequency)
        trusted_axis.grid(True)
        trusted_axis.set_axisbelow(True)
        if panel_index == 0:
            trusted_axis.legend(loc="upper right")
        annotation = trusted_axis.annotate(
            "",
            xy=(0, 0),
            xytext=(14, 14),
            textcoords="offset points",
            fontsize=8,
            color=theme.text,
            bbox={
                "boxstyle": "round,pad=0.4",
                "fc": theme.tooltip_background,
                "ec": theme.tooltip_border,
                "alpha": 0.96,
            },
            zorder=10,
        )
        annotation.set_visible(False)
        panels.append(HoverPanel(
            trusted_axis, axis_name, curve_frequency, curve_psd,
            hover_peaks, annotation,
        ))
    axes[-1].set_xlabel("Frequency, Hz")
    return panels


def attach_hover(fig, state: dict[str, Any]) -> None:
    """Tooltip at the curve point under the mouse, or at a marked peak."""

    def on_move(event) -> None:
        changed = False
        for panel in state["panels"]:
            if event.inaxes is not panel.ax or event.xdata is None:
                if panel.annotation.get_visible():
                    panel.annotation.set_visible(False)
                    changed = True
                continue
            nearest = None
            for frequency, psd, text in panel.peaks:
                x, y = panel.ax.transData.transform((frequency, psd))
                distance = float(np.hypot(x - event.x, y - event.y))
                if distance <= 12.0 and (nearest is None or distance < nearest[0]):
                    nearest = (distance, frequency, psd, text)
            if nearest is not None:
                _, frequency, psd, text = nearest
            else:
                index = int(np.argmin(np.abs(panel.frequency - event.xdata)))
                frequency = float(panel.frequency[index])
                psd = float(panel.psd[index])
                text = f"{panel.axis_name} {frequency:.3f} Hz\nPSD {psd:.3g} g²/Hz"
            right_side = event.x > fig.bbox.width * 0.7
            upper_half = event.y > panel.ax.bbox.y0 + 0.5 * panel.ax.bbox.height
            panel.annotation.xy = (frequency, psd)
            panel.annotation.set_text(text)
            panel.annotation.set_position(
                (-14 if right_side else 14, -14 if upper_half else 14),
            )
            panel.annotation.set_horizontalalignment("right" if right_side else "left")
            panel.annotation.set_verticalalignment("top" if upper_half else "bottom")
            panel.annotation.set_visible(True)
            changed = True
        if changed:
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)


def style_tk_toolbar(toolbar, window, theme: Theme) -> None:
    """Paint the Tk toolbar, its widgets and the window in ``theme``."""
    import tkinter as tk
    from tkinter import ttk

    def paint(widget, **options) -> None:
        for key, value in options.items():
            try:
                widget.configure({key: value})
            except tk.TclError:
                pass

    paint(toolbar, bg=theme.figure_background)
    if window is not None:
        paint(window, bg=theme.figure_background)
    for child in toolbar.winfo_children():
        paint(
            child,
            bg=theme.figure_background,
            fg=theme.text,
            activebackground=theme.axes_background,
            activeforeground=theme.text,
            selectcolor=theme.axes_background,
            highlightthickness=0,
        )
    # The Home and Save icons: matplotlib draws them light on a dark bar
    # when it is asked to repaint them.
    for button in getattr(toolbar, "_buttons", {}).values():
        try:
            toolbar._set_image_for_button(button)
        except Exception:
            pass
    style = ttk.Style(toolbar)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    style.configure(
        "TCombobox",
        fieldbackground=theme.axes_background,
        background=theme.figure_background,
        foreground=theme.text,
        arrowcolor=theme.text,
        bordercolor=theme.spine,
        lightcolor=theme.figure_background,
        darkcolor=theme.figure_background,
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", theme.axes_background)],
        foreground=[("readonly", theme.text)],
        selectbackground=[("readonly", theme.axes_background)],
        selectforeground=[("readonly", theme.text)],
    )
    toolbar.option_add("*TCombobox*Listbox.background", theme.axes_background)
    toolbar.option_add("*TCombobox*Listbox.foreground", theme.text)


def add_tk_controls(fig, labels: list[str], state: dict[str, Any], redraw) -> bool:
    """Run list, axis check boxes and theme list in the toolbar of a Tk window."""
    toolbar = getattr(fig.canvas.manager, "toolbar", None)
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError:
        return False
    if not isinstance(toolbar, tk.Widget):
        return False
    # The toolbar goes above the figure.
    toolbar.pack_forget()
    toolbar.pack(side=tk.TOP, fill=tk.X, before=fig.canvas.get_tk_widget())
    if len(labels) > 1:
        tk.Label(toolbar, text="   Run:").pack(side=tk.LEFT)
        choice = ttk.Combobox(
            toolbar,
            values=labels,
            state="readonly",
            width=max(len(label) for label in labels) + 2,
        )
        choice.current(state["run"])
        choice.pack(side=tk.LEFT)

        def on_select(_event) -> None:
            state["run"] = choice.current()
            redraw()

        choice.bind("<<ComboboxSelected>>", on_select)
    tk.Label(toolbar, text="   Axes:").pack(side=tk.LEFT)
    state["tk_variables"] = []
    for axis_name in AXIS_NAMES:
        variable = tk.BooleanVar(master=toolbar, value=state["shown"][axis_name])

        def toggle(name: str = axis_name, value=variable) -> None:
            state["shown"][name] = bool(value.get())
            redraw()

        tk.Checkbutton(
            toolbar, text=axis_name, variable=variable, command=toggle,
        ).pack(side=tk.LEFT)
        state["tk_variables"].append(variable)
    tk.Label(toolbar, text="   Theme:").pack(side=tk.LEFT)
    theme_names = list(THEMES)
    theme_choice = ttk.Combobox(
        toolbar,
        values=theme_names,
        state="readonly",
        width=max(len(name) for name in theme_names) + 2,
    )
    theme_choice.current(theme_names.index(state["theme"]))
    theme_choice.pack(side=tk.LEFT)
    window = getattr(fig.canvas.manager, "window", None)

    def on_theme(_event) -> None:
        state["theme"] = theme_names[theme_choice.current()]
        style_tk_toolbar(toolbar, window, THEMES[state["theme"]])
        redraw()

    theme_choice.bind("<<ComboboxSelected>>", on_theme)
    style_tk_toolbar(toolbar, window, THEMES[state["theme"]])
    return True


def show_trusted_viewer(
    runs: list[tuple[str, AnalysisResult, ApplicationConfig]],
    title: str,
) -> None:
    """One interactive window with figure 2.

    A list chooses the virtual run when there are several, check boxes show
    or hide the axes and the shown ones fill the window, a list switches the
    theme, and the mouse gives a tooltip with the frequency and, at a marked
    peak, its numbers. Without a Tk window the arrow keys change the run,
    x, y, z toggle the axes and t cycles the themes.
    """
    if plt.get_backend().lower() == "tkagg":
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

        # Only Home and Save: the window is for reading the figure.
        NavigationToolbar2Tk.toolitems = [
            item for item in NavigationToolbar2Tk.toolitems
            if item[0] in ("Home", "Save")
        ]
    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    manager = fig.canvas.manager
    if manager is not None:
        manager.set_window_title(title)
        window = getattr(manager, "window", None)
        if window is not None and hasattr(window, "state"):
            try:
                window.state("zoomed")  # Tk on Windows: open on the whole screen
            except Exception:
                pass
    state: dict[str, Any] = {
        "run": 0,
        "shown": {name: True for name in AXIS_NAMES},
        "panels": [],
        "theme": runs[0][2].visualization.theme,
    }

    def redraw() -> None:
        label, analysis_result, run_config = runs[state["run"]]
        state["panels"] = draw_trusted_figure(
            fig,
            analysis_result,
            run_config,
            tuple(name for name in AXIS_NAMES if state["shown"][name]),
            f"{title} — {label}" if label else title,
            THEMES[state["theme"]],
        )
        fig.canvas.draw_idle()

    redraw()
    attach_hover(fig, state)
    if not add_tk_controls(fig, [label for label, _, _ in runs], state, redraw):

        def on_key(event) -> None:
            if event.key in ("left", "right") and len(runs) > 1:
                step = 1 if event.key == "right" else -1
                state["run"] = (state["run"] + step) % len(runs)
                redraw()
            elif event.key in ("x", "y", "z"):
                name = event.key.upper()
                state["shown"][name] = not state["shown"][name]
                redraw()
            elif event.key == "t":
                names = list(THEMES)
                state["theme"] = names[(names.index(state["theme"]) + 1) % len(names)]
                redraw()

        fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


def build_analysis_figures(
    analysis_result: AnalysisResult,
    run_config: ApplicationConfig,
):
    visualization_data = analysis_result.visualization_data
    consolidated_frequency_regions = (
        analysis_result.consolidated_frequency_regions
    )
    analysis_bands = run_config.analysis_bands
    analysis_min_frequency, analysis_max_frequency = (
        get_analysis_frequency_limits(analysis_bands)
    )
    if visualization_data is None:
        raise RuntimeError("Visualization data was not built")

    theme = resolve_theme(run_config.visualization.theme)
    with plt.rc_context(theme_rc_params(theme)):
        stat_fig = draw_statistics_figure(
            visualization_data, analysis_bands, theme,
            analysis_min_frequency, analysis_max_frequency,
        )

    if consolidated_frequency_regions is None:
        raise RuntimeError("Consolidated frequency regions were not built")

    with plt.rc_context(theme_rc_params(theme)):
        trusted_fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    draw_trusted_figure(trusted_fig, analysis_result, run_config, theme=theme)

    return stat_fig, trusted_fig


def draw_statistics_figure(
    visualization_data: VisualizationData,
    analysis_bands: list[AnalysisBand],
    theme: Theme,
    analysis_min_frequency: float,
    analysis_max_frequency: float,
):
    """Figure 1: median PSD and stability per axis, in ``theme``."""
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

        draw_analysis_bands(psd_axis, analysis_bands, theme)
        curve_frequency, curve_psd, peak_tops = peak_tops_on_curve(
            axis_data.frequency,
            axis_data.median_psd,
            axis_data.peak_frequencies,
        )
        psd_axis.plot(
            curve_frequency,
            curve_psd,
            color=theme.series[axis_name],
            label=f"{axis_name} Median PSD",
        )
        peak_top_frequencies = np.array([top[0] for top in peak_tops])
        peak_top_values = np.array([top[1] for top in peak_tops])
        psd_axis.scatter(
            peak_top_frequencies,
            peak_top_values,
            color=theme.series[axis_name],
            marker="x",
            s=50,
            linewidths=1.6,
            zorder=4,
            label="Stable peaks",
        )
        annotate_peak_frequencies(
            psd_axis,
            peak_top_frequencies,
            peak_top_values,
            axis_data.peak_frequency_std_hz,
            axis_data.peak_local_snr_db,
        )
        psd_axis.set_title(f"{axis_name} axis — Median PSD")
        psd_axis.set_ylabel("PSD [g²/Hz]")
        psd_axis.set_xlim(analysis_min_frequency, analysis_max_frequency)
        psd_axis.grid(True)
        psd_axis.set_axisbelow(True)
        if axis_index == 0:
            psd_axis.legend(loc="upper right")

        stability_axis.plot(
            axis_data.frequency,
            axis_data.stability,
            color=theme.series[axis_name],
            label="Bin stability",
        )
        stability_axis.plot(
            axis_data.frequency,
            axis_data.local_window_power_stability,
            color=theme.secondary,
            linewidth=1.0,
            alpha=0.85,
            label="Local window power stability",
        )
        for band_index, band in enumerate(analysis_bands):
            stability_axis.hlines(
                band.min_stability,
                band.min_frequency,
                band.max_frequency,
                color=theme.accent,
                linestyle="--",
                linewidth=1.0,
                alpha=0.8,
                label=(
                    "Minimum window-power stability"
                    if band_index == 0
                    else None
                ),
            )
        stability_axis.scatter(
            axis_data.peak_frequencies,
            axis_data.peak_window_power_stability,
            color=theme.series[axis_name],
            marker="x",
            s=50,
            linewidths=1.6,
            zorder=4,
            label="Stable peaks — window power",
        )
        stability_axis.set_title(f"{axis_name} axis — Stability")
        stability_axis.set_ylabel("Mean / Std")
        stability_axis.set_xlim(
            analysis_min_frequency,
            analysis_max_frequency,
        )
        stability_axis.grid(True)
        stability_axis.set_axisbelow(True)
        if axis_index == 0:
            stability_axis.legend(loc="upper right")

    stat_axes[-1].set_xlabel("Frequency, Hz")
    stat_fig.suptitle("Statistical vibration analysis")
    return stat_fig



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
                config,
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
        analysis_result = analyze_sessions(sessions, config)
        aligned_psd = analysis_result.aligned_psd
        statistics = analysis_result.statistics
        peaks = analysis_result.peaks
        visualization_data = analysis_result.visualization_data
        frequency_cluster_diagnostics = (
            analysis_result.frequency_cluster_diagnostics
        )
        consolidated_frequency_regions = (
            analysis_result.consolidated_frequency_regions
        )
        append_run_diagnostics(
            run_result_paths.log,
            print_peak_candidate_diagnostics,
            analysis_result.candidate_diagnostics,
        )
        append_run_diagnostics(
            run_result_paths.log,
            print_session_frequency_clusters,
            frequency_cluster_diagnostics,
            analysis_bands,
            aligned_psd.x_stack.shape[0],
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
        append_run_diagnostics(
            run_result_paths.log,
            print_candidate_regions,
            consolidated_frequency_regions,
            config.visualization.trusted_frequency,
            aligned_psd.x_stack.shape[0],
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

    stat_fig, trusted_fig = build_analysis_figures(analysis_result, config)
    if run_result_paths is None:
        raise RuntimeError("Run result paths were not built")

    save_run_figures(run_result_paths, stat_fig, trusted_fig)
    print("Saved results:")
    print(f"  {run_result_paths.log}")
    print(f"  {run_result_paths.figure1}")
    print(f"  {run_result_paths.figure2}")
    if run_result_paths.raw.exists():
        print(f"  {run_result_paths.raw}")

    plt.close(stat_fig)
    plt.close(trusted_fig)
    if show_figures:
        # Figure 1 is only saved; figure 2 opens as the interactive window.
        show_trusted_viewer(
            [("", analysis_result, config)],
            f"{run_result_paths.log.stem} — {session_layout_name(config)}, "
            f"nperseg {config.welch.nperseg}",
        )

    return stop_requested


def replay_mode_directory(source_raw_path: Path, nperseg: int) -> Path:
    """replay_results/<record>/nperseg_<n>: modes never overwrite each other."""
    return REPLAY_RESULTS_DIRECTORY / source_raw_path.stem / f"nperseg_{nperseg}"


def build_replay_result_paths(
    source_raw_path: Path,
    virtual_mode: str,
    virtual_run_number: int,
    nperseg: int,
) -> RunResultPaths:
    result_directory = (
        replay_mode_directory(source_raw_path, nperseg)
        / virtual_mode
        / f"virtual_run{virtual_run_number:02d}"
    )
    result_directory.mkdir(parents=True, exist_ok=False)
    return RunResultPaths(
        log=result_directory / "result.txt",
        figure1=result_directory / "figure1.png",
        figure2=result_directory / "figure2.png",
        raw=source_raw_path,
    )


def initialize_replay_log(
    paths: RunResultPaths,
    source_raw_path: Path,
    virtual_mode: str,
    virtual_run_number: int,
    virtual_run_count: int,
    start_packet: int,
    end_packet: int,
    run_config: ApplicationConfig,
    sessions: list[SessionResult],
) -> None:
    measured_fs = float(np.mean([session.fs for session in sessions]))
    with paths.log.open("x", encoding="utf-8", newline="\n") as run_log:
        run_log.write(
            "Analysis source: raw replay\n"
            f"Source raw file: {source_raw_path.resolve()}\n"
            f"Virtual mode: {virtual_mode}\n"
            f"Virtual run: {virtual_run_number}/{virtual_run_count}\n"
            f"Packet range: {start_packet + 1}-{end_packet}\n"
            f"Packets/session: {run_config.session.packets_per_session}\n"
            f"Sessions: {len(sessions)}\n"
            f"ODR: {run_config.sensor.odr_hz:g} Hz\n"
            "Frequency tolerance: "
            f"{resolve_frequency_tolerance_hz(run_config.frequency_clustering, run_config.sensor.odr_hz):.2f} "
            "Hz\n"
            f"Welch nperseg: {run_config.welch.nperseg}\n"
            f"Welch noverlap: {run_config.welch.noverlap}\n"
            + trusted_threshold_line(run_config)
            + f"Fs={measured_fs:6.2f}\n\n"
        )


def save_replay_analysis(
    paths: RunResultPaths,
    analysis_result: AnalysisResult,
    run_config: ApplicationConfig,
) -> None:
    session_count = analysis_result.aligned_psd.x_stack.shape[0]
    append_run_diagnostics(
        paths.log,
        print_peak_candidate_diagnostics,
        analysis_result.candidate_diagnostics,
    )
    append_run_diagnostics(
        paths.log,
        print_session_frequency_clusters,
        analysis_result.frequency_cluster_diagnostics,
        run_config.analysis_bands,
        session_count,
    )
    append_run_diagnostics(
        paths.log,
        print_consolidated_frequency_regions,
        analysis_result.consolidated_frequency_regions,
        session_count,
    )
    append_run_diagnostics(
        paths.log,
        print_trusted_frequency_regions,
        analysis_result.consolidated_frequency_regions,
        run_config.visualization.trusted_frequency,
        session_count,
    )
    append_run_diagnostics(
        paths.log,
        print_candidate_regions,
        analysis_result.consolidated_frequency_regions,
        run_config.visualization.trusted_frequency,
        session_count,
    )
    stat_fig, trusted_fig = build_analysis_figures(
        analysis_result,
        run_config,
    )
    save_run_figures(paths, stat_fig, trusted_fig)
    plt.close(stat_fig)
    plt.close(trusted_fig)


def mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def build_replay_summary_rows(
    source_raw_path: Path,
    raw: RawMeasurement,
    virtual_mode: str,
    virtual_run_number: int,
    start_packet: int,
    end_packet: int,
    analysis_result: AnalysisResult,
    run_config: ApplicationConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    run_rows = []
    region_rows = []
    candidate_rows = []
    axis_values = (
        (
            "X",
            analysis_result.frequency_clusters.x,
            analysis_result.consolidated_frequency_regions.x,
        ),
        (
            "Y",
            analysis_result.frequency_clusters.y,
            analysis_result.consolidated_frequency_regions.y,
        ),
        (
            "Z",
            analysis_result.frequency_clusters.z,
            analysis_result.consolidated_frequency_regions.z,
        ),
    )
    packets_per_session = run_config.session.packets_per_session
    sessions_per_run = run_config.session.min_recommended_sessions
    packets_per_run = packets_per_session * sessions_per_run
    source_packet_count = raw.x.shape[0]
    virtual_run_count = source_packet_count // packets_per_run
    used_packet_count = virtual_run_count * packets_per_run
    packet_count = end_packet - start_packet
    duration_seconds = float(np.sum(
        raw.x.shape[1] / raw.packet_fs_hz[start_packet:end_packet]
    ))
    frequency_tolerance_hz = resolve_frequency_tolerance_hz(
        run_config.frequency_clustering,
        run_config.sensor.odr_hz,
    )
    support_total = analysis_result.aligned_psd.x_stack.shape[0]
    source_name = str(source_raw_path.resolve())
    common_values = {
        "source": source_name,
        "source_packet_count": source_packet_count,
        "used_packet_count": used_packet_count,
        "unused_packet_count": source_packet_count - used_packet_count,
        "odr_hz": run_config.sensor.odr_hz,
        "frequency_tolerance_hz": frequency_tolerance_hz,
        "mode": virtual_mode,
        "packets_per_session": packets_per_session,
        "sessions_per_run": sessions_per_run,
        "virtual_run": virtual_run_number,
        "packet_start": start_packet + 1,
        "packet_end": end_packet,
        "packet_count": packet_count,
        "duration_seconds": duration_seconds,
    }
    for axis_name, raw_clusters, consolidated_regions in axis_values:
        trusted_regions = [
            region
            for region in consolidated_regions
            if is_trusted_frequency_cluster(
                region,
                run_config.visualization.trusted_frequency,
            )
        ]
        run_rows.append({
            **common_values,
            "axis": axis_name,
            "raw_clusters": len(raw_clusters),
            "consolidated_regions": len(consolidated_regions),
            "trusted_regions": len(trusted_regions),
            "sources_ge_2": sum(
                len(region.source_clusters) >= 2
                for region in consolidated_regions
            ),
            "mean_support_fraction": mean_or_none([
                region.support_fraction
                for region in consolidated_regions
            ]),
            "mean_sigma_f_hz": mean_or_none([
                cluster.frequency_std_hz
                for cluster in raw_clusters
            ]),
            "mean_raw_cluster_span_hz": mean_or_none([
                cluster.maximum_frequency - cluster.minimum_frequency
                for cluster in raw_clusters
            ]),
        })
        for region in trusted_regions:
            evidence = region.median_evidence
            region_rows.append({
                **common_values,
                "axis": axis_name,
                "band": region.band_name,
                "freq_hz": region.frequency,
                "med_freq_hz": reported_median_frequency(evidence),
                "med_freq_bin_hz": evidence.peak_frequency,
                "med_top_rise_db": evidence.refined_rise_db,
                "support_n": region.support_count,
                "support_total": support_total,
                "support_fraction": region.support_fraction,
                "range_min_hz": region.minimum_frequency,
                "range_max_hz": region.maximum_frequency,
                "frequency_std_hz": max(
                    source.cluster.frequency_std_hz
                    for source in region.source_clusters
                ),
                "med_prom_db": evidence.prominence_db,
                "med_contrast_db": evidence.local_contrast_db,
                "band_contrast_db": evidence.band_contrast_db,
                "sources": len(region.source_clusters),
                "weight": get_trusted_frequency_weight(
                    region,
                    run_config.visualization.trusted_frequency,
                ),
            })
        for region in select_candidate_regions(
            consolidated_regions,
            run_config.visualization.trusted_frequency,
        ):
            evidence = region.median_evidence
            candidate_rows.append({
                **common_values,
                "axis": axis_name,
                "band": region.band_name,
                "freq_hz": region.frequency,
                "med_freq_hz": reported_median_frequency(evidence),
                "med_freq_bin_hz": evidence.peak_frequency,
                "support_n": region.support_count,
                "support_total": support_total,
                "support_fraction": region.support_fraction,
                "med_prom_db": evidence.prominence_db,
                "threshold_db": (
                    run_config.visualization.trusted_frequency
                    .min_median_prominence_db
                ),
                "range_min_hz": region.minimum_frequency,
                "range_max_hz": region.maximum_frequency,
            })
    return run_rows, region_rows, candidate_rows


def write_csv_rows(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    with path.open("x", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_replay_summaries(
    result_root: Path,
    source_raw_path: Path,
    raw: RawMeasurement,
    run_rows: list[dict[str, Any]],
    region_rows: list[dict[str, Any]],
    layout_rows: list[dict[str, Any]],
    base_config: ApplicationConfig,
    candidate_rows: list[dict[str, Any]] | None = None,
) -> None:
    run_fields = [
        "source", "source_packet_count", "used_packet_count",
        "unused_packet_count", "odr_hz", "frequency_tolerance_hz",
        "mode", "packets_per_session", "sessions_per_run", "virtual_run",
        "packet_start", "packet_end", "packet_count", "duration_seconds",
        "axis", "raw_clusters", "consolidated_regions", "trusted_regions",
        "sources_ge_2", "mean_support_fraction", "mean_sigma_f_hz",
        "mean_raw_cluster_span_hz",
    ]
    region_fields = [
        "source", "source_packet_count", "used_packet_count",
        "unused_packet_count", "odr_hz", "frequency_tolerance_hz",
        "mode", "packets_per_session", "sessions_per_run", "virtual_run",
        "packet_start", "packet_end", "packet_count", "duration_seconds",
        "axis", "band", "freq_hz", "med_freq_hz", "med_freq_bin_hz",
        "med_top_rise_db", "support_n",
        "support_total", "support_fraction", "range_min_hz", "range_max_hz",
        "frequency_std_hz", "med_prom_db", "med_contrast_db",
        "band_contrast_db", "sources", "weight",
    ]
    write_csv_rows(result_root / "replay_runs.csv", run_rows, run_fields)
    write_csv_rows(
        result_root / "replay_regions.csv",
        region_rows,
        region_fields,
    )
    candidate_fields = [
        *region_fields[:region_fields.index("axis")],
        "axis", "band", "freq_hz", "med_freq_hz",
        "med_freq_bin_hz", "support_n", "support_total", "support_fraction",
        "med_prom_db", "threshold_db",
        "range_min_hz", "range_max_hz",
    ]
    write_csv_rows(
        result_root / "replay_candidates.csv",
        candidate_rows or [],
        candidate_fields,
    )
    effective_tolerance = resolve_frequency_tolerance_hz(
        base_config.frequency_clustering,
        raw.requested_odr_hz,
    )
    with (result_root / "replay_metadata.txt").open(
        "x",
        encoding="utf-8",
        newline="\n",
    ) as metadata_file:
        metadata_file.write(
            f"Source raw: {source_raw_path.resolve()}\n"
            f"Raw packets: {raw.x.shape[0]}\n"
            f"ODR: {raw.requested_odr_hz:g} Hz\n"
            f"Effective frequency tolerance: {effective_tolerance:.2f} Hz\n"
            f"Old detector mode: Welch nperseg {base_config.welch.nperseg}, "
            f"noverlap {base_config.welch.noverlap}\n"
            "Region frequency_std_hz: maximum source-cluster sigma_f\n"
            "\nLayouts:\n"
        )
        thresholds = dict(active_old_detector_mode(base_config).thresholds)
        for layout in layout_rows:
            metadata_file.write(
                f"\n{layout['mode']}:\n"
                f"  Med.Prom threshold: {thresholds[layout['mode']]:g} dB\n"
                f"  packets/session: {layout['packets_per_session']}\n"
                f"  sessions/run: {layout['sessions_per_run']}\n"
                f"  packets/run: {layout['packets_per_run']}\n"
                f"  virtual runs: {layout['virtual_runs']}\n"
                f"  used packets: {layout['used_packet_count']}\n"
                f"  unused packets: {layout['unused_packet_count']}\n"
            )


def show_replay_layout(
    source_raw_path: Path,
    layout: str,
    base_config: ApplicationConfig,
) -> None:
    """Analyse every virtual run of one layout and open figure 2 of them in
    one interactive window; nothing is written."""
    raw = load_raw_measurement(source_raw_path)
    packets_per_session, target_sessions = parse_layout(layout)
    run_slices = build_virtual_run_slices(
        raw.x.shape[0], packets_per_session, target_sessions,
    )
    if not run_slices:
        raise ValueError(f"{source_raw_path.name} is too short for layout {layout}")
    replay_config = build_replay_config(
        base_config, raw, packets_per_session, target_sessions,
    )
    runs = []
    for index, (start_packet, end_packet) in enumerate(run_slices, start=1):
        sessions = build_sessions_from_raw(
            raw, start_packet, end_packet, replay_config,
        )
        runs.append((
            f"run {index}/{len(run_slices)}, packets {start_packet + 1}–{end_packet}",
            analyze_sessions(sessions, replay_config),
            replay_config,
        ))
    show_trusted_viewer(
        runs,
        f"{source_raw_path.stem} — {layout}, nperseg {replay_config.welch.nperseg}",
    )


def replay_raw_measurement(
    source_raw_path: Path,
    virtual_modes: list[str],
    base_config: ApplicationConfig,
) -> None:
    modes = resolve_replay_layouts(virtual_modes, base_config)
    raw = load_raw_measurement(source_raw_path)
    replay_run_rows = []
    replay_region_rows = []
    replay_candidate_rows = []
    layout_rows = []
    result_root = replay_mode_directory(
        source_raw_path, base_config.welch.nperseg,
    )
    result_root.mkdir(parents=True, exist_ok=False)
    for mode in modes:
        packets_per_session, target_sessions = parse_layout(mode)
        packets_per_run = packets_per_session * target_sessions
        run_slices = build_virtual_run_slices(
            raw.x.shape[0],
            packets_per_session,
            target_sessions,
        )
        used_packet_count = len(run_slices) * packets_per_run
        layout_rows.append({
            "mode": mode,
            "packets_per_session": packets_per_session,
            "sessions_per_run": target_sessions,
            "packets_per_run": packets_per_run,
            "virtual_runs": len(run_slices),
            "used_packet_count": used_packet_count,
            "unused_packet_count": raw.x.shape[0] - used_packet_count,
        })
        if not run_slices:
            print(f"Skipping replay layout {mode}: not enough packets")
            continue
        replay_config = build_replay_config(
            base_config,
            raw,
            packets_per_session,
            target_sessions,
        )
        for virtual_run_index, (start_packet, end_packet) in enumerate(
            run_slices,
            start=1,
        ):
            sessions = build_sessions_from_raw(
                raw,
                start_packet,
                end_packet,
                replay_config,
            )
            analysis_result = analyze_sessions(sessions, replay_config)
            paths = build_replay_result_paths(
                source_raw_path,
                mode,
                virtual_run_index,
                replay_config.welch.nperseg,
            )
            initialize_replay_log(
                paths,
                source_raw_path,
                mode,
                virtual_run_index,
                len(run_slices),
                start_packet,
                end_packet,
                replay_config,
                sessions,
            )
            save_replay_analysis(paths, analysis_result, replay_config)
            run_rows, region_rows, candidate_rows = build_replay_summary_rows(
                source_raw_path,
                raw,
                mode,
                virtual_run_index,
                start_packet,
                end_packet,
                analysis_result,
                replay_config,
            )
            replay_run_rows.extend(run_rows)
            replay_region_rows.extend(region_rows)
            replay_candidate_rows.extend(candidate_rows)
            print(
                f"Saved replay {mode} run "
                f"{virtual_run_index}/{len(run_slices)}: "
                f"{paths.log.parent}"
            )
    save_replay_summaries(
        result_root,
        source_raw_path,
        raw,
        replay_run_rows,
        replay_region_rows,
        layout_rows,
        base_config,
        replay_candidate_rows,
    )


try:
    if cli_arguments.replay is not None:
        if total_runs != 1:
            raise ValueError("--repeat cannot be used with --replay")
        if cli_arguments.show:
            show_replay_layout(
                cli_arguments.replay,
                cli_arguments.virtual_mode[0],
                config,
            )
        else:
            replay_raw_measurement(
                cli_arguments.replay,
                cli_arguments.virtual_mode,
                config,
            )
    else:
        for run_number in range(1, total_runs + 1):
            stop_series = run_measurement(
                run_number,
                total_runs,
                show_figures=(total_runs == 1),
            )
            if stop_series:
                break
finally:
    if ser is not None:
        ser.close()
