import struct
from dataclasses import dataclass

import serial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons

from scipy.signal import welch
from scipy.signal import find_peaks

# ==========================================================
# Настройки
# ==========================================================

PORT = "COM8"
BAUD = 115200

# AXIS = "X"          # X / Y / Z

MAGIC = b"VIB2"
HEADER_SIZE = 10

# Welch
NPERSEG = 1024
NOVERLAP = 512
PACKETS_PER_SESSION = 8

# Диапазоны анализа
FFT_MIN_FREQ = 0.5
FFT_MAX_FREQ = 20
PSD_MIN_FREQ = 0.5
PSD_MAX_FREQ = 20

# Поиск пиков FFT
FFT_THRESHOLD = 2.1        # пик должен быть выше среднего уровня в 2.1 раза

# Поиск пиков PSD
PROMINENCE = 0.3          # 30% от максимума
MIN_DISTANCE_HZ = 1.0

COLORS = {
    "X": "tab:blue",
    "Y": "tab:orange",
    "Z": "tab:green",
}

# ==========================================================


@dataclass
class FFTResult:
    freq: np.ndarray
    amplitude: np.ndarray
    resolution: float
    peaks: np.ndarray
    peak_freq: float
    peak_amplitude: float


@dataclass
class AverageFFTResult:
    freq: np.ndarray
    amplitude: np.ndarray


@dataclass
class PSDResult:
    freq: np.ndarray
    psd: np.ndarray
    resolution: float
    peaks: np.ndarray
    peak_freq: float
    peak_psd: float


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


# ==========================================================

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

    # peak_index = np.argmax(amplitude)
    '''mask = (freq >= FFT_MIN_FREQ) & (freq <= FFT_MAX_FREQ)
    peak_local = np.argmax(amplitude[mask])
    peak_index = np.where(mask)[0][peak_local]'''
    
    mask = (freq >= FFT_MIN_FREQ) & (freq <= FFT_MAX_FREQ)
    freq_search = freq[mask]
    amp_search = amplitude[mask]
    
    distance = max(int(MIN_DISTANCE_HZ / resolution), 1)
    peaks, _ = find_peaks(
        amp_search,
        prominence=np.mean(amp_search) * FFT_THRESHOLD,
        distance=distance
    )
    if len(peaks):
        dominant = peaks[np.argmax(amp_search[peaks])]
        peak_freq = freq_search[dominant]
        peak_amplitude = amp_search[dominant]
    else:
        peak_freq = np.nan
        peak_amplitude = np.nan
    return FFTResult(
        freq=freq,
        amplitude=amplitude,
        resolution=resolution,
        peaks=np.where(mask)[0][peaks],
        peak_freq=peak_freq,
        peak_amplitude=peak_amplitude
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

    distance = max(int(MIN_DISTANCE_HZ / resolution), 1)

    peaks, properties = find_peaks(
        psd,
        prominence=np.max(psd) * PROMINENCE,
        distance=distance
    )

    if len(peaks):

        dominant = peaks[np.argmax(psd[peaks])]

        peak_freq = freq[dominant]
        peak_psd = psd[dominant]

    else:

        peak_freq = np.nan
        peak_psd = np.nan

    return PSDResult(
        freq=freq,
        psd=psd,
        resolution=resolution,
        peaks=peaks,
        peak_freq=peak_freq,
        peak_psd=peak_psd
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

signals = {
    "X": [],
    "Y": [],
    "Z": [],
}
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

    signals["X"].append(x)
    signals["Y"].append(y)
    signals["Z"].append(z)

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
    
    packets = len(signals["X"])
    print(
        f"\rPackets: {packets:4d}"
        f"   Duration: {packets * len(x) / np.mean(fs_list):6.1f} s"
        f"   Fs={np.mean(fs_list):6.2f}",
        end=""
    )

        continue

    if packet is None:
        continue

    fs, x, y, z = packet

    signals["X"].append(x)
    signals["Y"].append(y)
    signals["Z"].append(z)

    current_session["X"].append(x)
    current_session["Y"].append(y)
    current_session["Z"].append(z)
    current_session_fs.append(fs)

    fs_list.append(fs)

    session_packets = len(current_session["X"])

    if session_packets == PACKETS_PER_SESSION:
        sessions.append({
            "signals": current_session,
            "fs": current_session_fs,
        })
        current_session = {
            "X": [],
            "Y": [],
            "Z": [],
        }
        current_session_fs = []
    
    packets = len(signals["X"])
    print(
        f"\rPackets: {packets:4d}"
        f"   Duration: {packets * len(x) / np.mean(fs_list):6.1f} s"
        f"   Fs={np.mean(fs_list):6.2f}",
        end=""
    )

    if stop_requested and session_packets == PACKETS_PER_SESSION:
        break

print()

# ==========================================================

# signal = np.concatenate(signals)
signals = {
    axis: np.concatenate(data)
    for axis, data in signals.items()
}

print()
for axis, signal in signals.items():
    print(
        axis,
        signal.min(),
        signal.max(),
        np.mean(signal)
    )

fs = np.mean(fs_list)

duration = len(signals["X"]) / fs

print()
print("--------------------------------------")
print(f"Samples   : {len(signals['X'])}")
print(f"Duration  : {duration:.2f} s")
print(f"Fs        : {fs:.3f} Hz")
print("--------------------------------------")

# ==========================================================
# Предобработка
# ==========================================================

# signal = signal - np.mean(signal)

# ==========================================================
# FFT и PSD
# ==========================================================

'''fft = compute_fft(signal, fs)
avg_freq, avg_amp = compute_average_fft(signal, fs)
psd = compute_psd(signal, fs)'''

results = {}

for axis, signal in signals.items():

    signal = signal - np.mean(signal)

    fft = compute_fft(signal, fs)

    avg = compute_average_fft(signal, fs)

    psd = compute_psd(signal, fs)

    results[axis] = {
        "signal": signal,
        "fft": fft,
        "avg": avg,
        "psd": psd,
    }

'''axis = "Y"

signal = results[axis]["signal"]

fft = results[axis]["fft"]

avg = results[axis]["avg"]

psd = results[axis]["psd"]
# ==========================================================

print()
print("Measurement summary")
print("--------------------------------------------------")
print(f"Axis            : {axis}")
print(f"Duration        : {duration:.2f} s")
print(f"Samples         : {len(signal)}")
print(f"Sampling rate   : {fs:.3f} Hz")
print(f"FFT resolution  : {fft.resolution:.4f} Hz")
print(f"PSD resolution  : {psd.resolution:.4f} Hz")

if len(psd.peaks):

    print()
    print("Dominant peak")
    print("--------------------------------------------------")
    print(f"FFT frequency   : {fft.peak_freq:.3f} Hz")
    print(f"FFT amplitude   : {fft.peak_amplitude:.4f} g")
    print(f"PSD frequency   : {psd.peak_freq:.3f} Hz")
    print(f"PSD             : {psd.peak_psd:.3e} g²/Hz")

    print()
    print("Detected peaks")
    print("--------------------------------------------------")

    for p in psd.peaks:

        print(
            f"{psd.freq[p]:7.2f} Hz    PSD={psd.psd[p]:.3e}"
        )

else:

    print()

    print("No significant peaks detected.")'''
    
print()
print("Measurement summary")
print("==================================================")
print(f"Duration      : {duration:.2f} s")
print(f"Sampling rate : {fs:.3f} Hz")

for axis, result in results.items():

    signal = result["signal"]
    fft = result["fft"]
    psd = result["psd"]

    print()
    print(f"Axis {axis}")
    print("--------------------------------------------------")
    print(f"Samples         : {len(signal)}")
    print(f"FFT resolution  : {fft.resolution:.4f} Hz")
    print(f"PSD resolution  : {psd.resolution:.4f} Hz")

    if len(psd.peaks):

        print(f"FFT peak        : {fft.peak_freq:.3f} Hz")
        print(f"FFT amplitude   : {fft.peak_amplitude:.4f} g")
        print(f"PSD peak        : {psd.peak_freq:.3f} Hz")
        print(f"PSD             : {psd.peak_psd:.3e} g²/Hz")

        print("Detected PSD peaks:")

        for p in psd.peaks:
            print(
                f"  {psd.freq[p]:7.2f} Hz    PSD={psd.psd[p]:.3e}"
            )

    else:

        print("No significant peaks.")

# ==========================================================
# Графики
# ==========================================================

fig, ax = plt.subplots(4, 1, figsize=(14, 12))
fig.subplots_adjust(
    left=0.08,
    right=0.84,
    top=0.93,
    bottom=0.06,
    hspace=0.45
)
#check_ax = plt.axes([0.87, 0.80, 0.08, 0.14])
check_ax = plt.axes([0.80, 0.78, 0.08, 0.13])

check = CheckButtons(
    check_ax,
    ["X", "Y", "Z"],
    [True, True, True]
)
for label, axis in zip(check.labels, ["X", "Y", "Z"]):
    label.set_color(COLORS[axis])
    label.set_fontweight("bold")
check_ax.set_frame_on(False)
check_ax.set_facecolor("none")
for label in check.labels:
    label.set_fontsize(11)

lines = {
    "time": {},
    "fft": {},
    "avg": {},
    "psd": {},
}

markers = {
    "fft": {},
    "psd": {},
}

annotations = {
    "fft": {},
    "psd": {},
}

# ----------------------------------------------------------
# Временной сигнал
# ----------------------------------------------------------

'''t = np.arange(len(signal)) / fs

ax[0].plot(t, signal, linewidth=1)

ax[0].set_title("Acceleration")

ax[0].set_xlabel("Time [s]")

ax[0].set_ylabel("g")

ax[0].grid(True)'''

for axis, result in results.items():

    signal = result["signal"]

    t = np.arange(len(signal)) / fs

    line, = ax[0].plot(
        t,
        signal,
        color=COLORS[axis],
        label=axis,
        linewidth=1
    )

    lines["time"][axis] = line

# ax[0].legend()

# ----------------------------------------------------------
# FFT
# ----------------------------------------------------------

'''mask_fft = fft.freq <= FFT_MAX_FREQ

ax[1].plot(
    fft.freq[mask_fft],
    fft.amplitude[mask_fft]
)'''

for axis, result in results.items():

    fft = result["fft"]

    mask_fft = fft.freq <= FFT_MAX_FREQ

    line, = ax[1].plot(
        fft.freq[mask_fft],
        fft.amplitude[mask_fft],
        color=COLORS[axis],
        label=axis
    )

    lines["fft"][axis] = line

    markers["fft"][axis] = []
    annotations["fft"][axis] = []

    for p in fft.peaks:

        marker, = ax[1].plot(
            fft.freq[p],
            fft.amplitude[p],
            "o",
            color=COLORS[axis]
        )

        text = ax[1].annotate(
            f"{fft.freq[p]:.2f} Hz",
            (fft.freq[p], fft.amplitude[p]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=COLORS[axis]
        )

        markers["fft"][axis].append(marker)
        annotations["fft"][axis].append(text)

# ax[1].legend()
ax[1].set_xlim(0, FFT_MAX_FREQ)
ax[1].grid(True)
ax[1].set_title("FFT")
ax[1].set_xlabel("Frequency [Hz]")
ax[1].set_ylabel("Amplitude [g]")

# FFT_SMOOTH

'''mask_avg = avg.freq <= FFT_MAX_FREQ

ax[2].plot(
    avg.freq[mask_avg],
    avg.amplitude[mask_avg]
)'''

for axis, result in results.items():

    avg = result["avg"]

    mask_avg = avg.freq <= FFT_MAX_FREQ

    line, = ax[2].plot(
        avg.freq[mask_avg],
        avg.amplitude[mask_avg],
        color=COLORS[axis],
        label=axis
    )

    lines["avg"][axis] = line

ax[2].legend()

ax[2].set_xlim(0, FFT_MAX_FREQ)
ax[2].grid(True)
ax[2].set_title("Average FFT")
ax[2].set_xlabel("Frequency [Hz]")
ax[2].set_ylabel("Amplitude [g]")

# ----------------------------------------------------------
# PSD
# ----------------------------------------------------------

#ax[3].semilogy(psd.freq, psd.psd)

for axis, result in results.items():

    psd = result["psd"]

    line, = ax[3].semilogy(
        psd.freq,
        psd.psd,
        color=COLORS[axis],
        label=axis
    )

    lines["psd"][axis] = line

    markers["psd"][axis] = []
    annotations["psd"][axis] = []

    for p in psd.peaks:

        marker, = ax[3].plot(
            psd.freq[p],
            psd.psd[p],
            "o",
            color=COLORS[axis]
        )

        text = ax[3].annotate(
            f"{psd.freq[p]:.2f} Hz",
            (psd.freq[p], psd.psd[p]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=COLORS[axis]
        )

        markers["psd"][axis].append(marker)
        annotations["psd"][axis].append(text)

ax[3].legend()
ax[3].set_xlim(PSD_MIN_FREQ, PSD_MAX_FREQ)
ax[3].grid(True, which="major", alpha=0.6)
ax[3].grid(True, which="minor", alpha=0.2)
ax[3].set_title("PSD (Welch)")
ax[3].set_xlabel("Frequency [Hz]")
ax[3].set_ylabel("PSD [g²/Hz]")

# ----------------------------------------------------------

def toggle_axis(label):

    visible = not lines["time"][label].get_visible()

    for group in lines.values():
        group[label].set_visible(visible)

    for group in markers.values():
        for marker in group[label]:
            marker.set_visible(visible)

    for group in annotations.values():
        for text in group[label]:
            text.set_visible(visible)

    fig.canvas.draw_idle()

check.on_clicked(toggle_axis)

fig.suptitle(
    f"Fs={fs:.2f} Hz    "
    f"Duration={duration:.1f} s    "
    f"FFT Δf={results['X']['fft'].resolution:.4f} Hz    "
    f"PSD Δf={results['X']['psd'].resolution:.4f} Hz"
)


plt.show()