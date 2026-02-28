"""
core/signal_processing.py
Limpieza y filtrado de señales de acelerómetro.
Separado del análisis para poder reutilizarse en cualquier contexto.
"""
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import uniform_filter1d, gaussian_filter1d

SAMPLE_RATE = 100  # Hz default


def estimate_sample_rate(df: pd.DataFrame) -> int:
    """Estima la tasa de muestreo real desde los timestamps."""
    if "time" in df.columns and len(df) > 10:
        dt = np.median(np.diff(df["time"].values))
        if dt > 0:
            return max(1, round(1.0 / dt))
    return SAMPLE_RATE


def normalize_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza la columna de tiempo a segundos desde cero.
    Maneja nanosegundos, segundos_elapsed, y timestamps Unix.
    """
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    if "seconds_elapsed" in df.columns:
        t = pd.to_numeric(df["seconds_elapsed"], errors="coerce")
        df["time"] = t - t.iloc[0]
    else:
        time_col = next((c for c in df.columns if "time" in c), None)
        if time_col:
            raw = pd.to_numeric(df[time_col], errors="coerce")
            if raw.median() > 1e12:  # nanosegundos → segundos
                raw = raw / 1e9
            df["time"] = raw - raw.iloc[0]

    return df


def normalize_axes(df: pd.DataFrame) -> pd.DataFrame:
    """Mapea columnas de ejes al estándar x, y, z."""
    df = df.copy()
    axis_map = {}
    for col in df.columns:
        cl = col.lower()
        if cl in ["x", "accel_x", "acceleration x (m/s^2)", "ax"]:
            axis_map[col] = "x"
        elif cl in ["y", "accel_y", "acceleration y (m/s^2)", "ay"]:
            axis_map[col] = "y"
        elif cl in ["z", "accel_z", "acceleration z (m/s^2)", "az"]:
            axis_map[col] = "z"
    df.rename(columns=axis_map, inplace=True)
    for axis in ["x", "y", "z"]:
        if axis in df.columns:
            df[axis] = pd.to_numeric(df[axis], errors="coerce")
    return df


def remove_outliers(series: np.ndarray, z_thresh: float = 4.0) -> np.ndarray:
    """
    Reemplaza outliers extremos por interpolación lineal.
    Outlier = valor a más de z_thresh desviaciones estándar.
    """
    s = series.copy().astype(float)
    mean, std = np.mean(s), np.std(s)
    if std == 0:
        return s
    mask = np.abs(s - mean) > z_thresh * std
    idx = np.arange(len(s))
    s[mask] = np.interp(idx[mask], idx[~mask], s[~mask])
    return s


def butter_lowpass(data: np.ndarray, cutoff: float, fs: int, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    cutoff = min(cutoff, nyq * 0.99)
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, data)


def butter_bandpass(data: np.ndarray, low: float, high: float, fs: int, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    lo = low / nyq
    hi = min(high / nyq, 0.99)
    b, a = butter(order, [lo, hi], btype="band")
    return filtfilt(b, a, data)


def preprocess_accelerometer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline completo de limpieza para datos de acelerómetro:
    1. Normalizar tiempo y ejes
    2. Eliminar outliers
    3. Filtro pasa-bajos (quita ruido >20 Hz)
    4. Calcular magnitud sin gravedad
    """
    df = normalize_time(df)
    df = normalize_axes(df)
    df.dropna(subset=["time"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    fs = estimate_sample_rate(df)
    df["_fs"] = fs

    for axis in ["x", "y", "z"]:
        if axis in df.columns:
            clean = remove_outliers(df[axis].fillna(0).values)
            df[axis] = clean
            df[f"{axis}_filt"] = butter_lowpass(clean, cutoff=min(20.0, fs / 2 - 1), fs=fs)

    if all(f"{a}_filt" in df.columns for a in ["x", "y", "z"]):
        df["magnitude"] = np.sqrt(
            df["x_filt"] ** 2 +
            df["y_filt"] ** 2 +
            (df["z_filt"] - 9.81) ** 2
        )

    return df


def detect_steps(accel: pd.DataFrame):
    """
    Detección de pisadas robusta con filtro pasa-banda 1.5–4 Hz.
    Devuelve (peak_indices, peak_times, peak_values)
    """
    fs = int(accel["_fs"].iloc[0]) if "_fs" in accel.columns else SAMPLE_RATE

    if "x_filt" in accel.columns:
        raw = np.sqrt(
            accel["x_filt"].values ** 2 +
            accel["y_filt"].values ** 2 +
            accel["z_filt"].values ** 2
        )
    elif "magnitude" in accel.columns:
        raw = accel["magnitude"].values
    else:
        raw = accel["z"].values

    signal_bp = butter_bandpass(raw, low=1.5, high=4.0, fs=fs)
    envelope = uniform_filter1d(np.abs(signal_bp), size=int(fs * 0.1))

    min_dist = int(fs * 0.27)
    threshold = np.percentile(envelope, 65)
    prominence = np.std(envelope) * 0.5

    peaks, _ = find_peaks(envelope, height=threshold, distance=min_dist, prominence=prominence)

    if len(peaks) < 4:
        peaks, _ = find_peaks(
            envelope,
            height=np.mean(envelope),
            distance=min_dist,
            prominence=np.std(envelope) * 0.2
        )

    accel["step_signal"] = signal_bp
    accel["step_envelope"] = envelope

    times = accel["time"].values[peaks]
    values = np.abs(signal_bp[peaks])

    return peaks, times, values


def cadence_over_time(peak_times: np.ndarray, window_size: int = 40):
    """Cadencia suavizada con ventana deslizante."""
    if len(peak_times) < 5:
        return np.array([]), np.array([])

    if len(peak_times) < window_size + 1:
        window_size = max(10, len(peak_times) // 3)

    intervals = np.diff(peak_times)
    times_mid = peak_times[1:]
    cadences, t_out = [], []

    for i in range(len(intervals) - window_size + 1):
        win = intervals[i:i + window_size]
        valid = (win >= 0.27) & (win <= 1.0)
        if valid.sum() > window_size * 0.5:
            cadences.append(60.0 / np.median(win[valid]))
            t_out.append(times_mid[i + window_size // 2])

    if len(cadences) < 3:
        return np.array([]), np.array([])

    t_arr = np.array(t_out)
    c_arr = gaussian_filter1d(np.array(cadences), sigma=6)
    c_arr = np.clip(c_arr, 100, 230)

    return t_arr, c_arr
