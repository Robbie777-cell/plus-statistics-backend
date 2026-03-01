import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import gaussian_filter1d


class SignalProcessor:
    def __init__(self):
        self.fs = 100  # frecuencia de muestreo Hz
        self.lowpass_cutoff = 20
        self.bandpass_low = 0.5
        self.bandpass_high = 5.0
        self.outlier_threshold = 3

    def process_accelerometer(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._normalize_columns(df)
        df = self._normalize_time(df)
        df = self._remove_outliers(df)
        df = self._apply_filters(df)
        return df

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        col_map = {}
        for col in df.columns:
            lower = col.lower().strip()
            if lower in ["time", "timestamp", "t", "seconds", "time (s)"]:
                col_map[col] = "time"
            elif lower in ["x", "x-axis", "ax", "accel x", "acceleration x"]:
                col_map[col] = "x"
            elif lower in ["y", "y-axis", "ay", "accel y", "acceleration y"]:
                col_map[col] = "y"
            elif lower in ["z", "z-axis", "az", "accel z", "acceleration z"]:
                col_map[col] = "z"
        df = df.rename(columns=col_map)
        for axis in ["x", "y", "z"]:
            if axis not in df.columns:
                df[axis] = 0.0
        if "time" not in df.columns:
            df["time"] = np.arange(len(df)) / self.fs
        return df[["time", "x", "y", "z"]]

    def _normalize_time(self, df: pd.DataFrame) -> pd.DataFrame:
        df["time"] = pd.to_numeric(df["time"], errors="coerce")
        df = df.dropna(subset=["time"])
        df = df.sort_values("time").reset_index(drop=True)
        df["time"] = df["time"] - df["time"].iloc[0]
        return df

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        for axis in ["x", "y", "z"]:
            df[axis] = pd.to_numeric(df[axis], errors="coerce")
            mean = df[axis].mean()
            std = df[axis].std()
            if std > 0:
                z_scores = np.abs((df[axis] - mean) / std)
                df.loc[z_scores > self.outlier_threshold, axis] = mean
        df = df.dropna().reset_index(drop=True)
        return df

    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        n_samples = len(df)
        if n_samples < 10:
            return df
        try:
            nyq = self.fs / 2
            low = self.lowpass_cutoff / nyq
            b, a = signal.butter(4, low, btype="low")
            for axis in ["x", "y", "z"]:
                df[axis] = signal.filtfilt(b, a, df[axis].values)
        except Exception:
            pass
        return df

    def detect_steps(self, df: pd.DataFrame):
        try:
            magnitude = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)
            smoothed = gaussian_filter1d(magnitude.values, sigma=3)
            threshold = smoothed.mean() + smoothed.std() * 0.3
            min_distance = int(self.fs * 0.25)
            peaks, _ = signal.find_peaks(smoothed, height=threshold, distance=min_distance)
            return peaks
        except Exception:
            return np.array([])

    def compute_cadence_over_time(self, df: pd.DataFrame, peaks: np.ndarray):
        if len(peaks) < 4:
            return np.array([]), np.array([])
        try:
            peak_times = df["time"].iloc[peaks].values
            window_size = max(10, len(peak_times) // 5)
            if len(peak_times) < window_size + 1:
                window_size = max(2, len(peak_times) // 3)
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
        except Exception:
            return np.array([]), np.array([])
