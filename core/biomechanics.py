import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from .signal_processing import SignalProcessor


class BiomechanicsAnalyzer:
    def __init__(self):
        self.processor = SignalProcessor()

    def analyze(self, df: pd.DataFrame, loc_df: pd.DataFrame = None) -> dict:
        try:
            peaks = self.processor.detect_steps(df)
            n_steps = len(peaks)
            duration = float(df["time"].iloc[-1] - df["time"].iloc[0]) if len(df) > 1 else 0
            duration_min = duration / 60.0

            # Cadencia
            cadence_avg = 0.0
            if n_steps > 0 and duration > 0:
                cadence_avg = (n_steps / duration) * 60.0
                cadence_avg = float(np.clip(cadence_avg, 100, 230))

            # Magnitud de impacto
            magnitude = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)
            ground_shock = float(magnitude.mean())

            # Asimetría
            left = magnitude.iloc[::2].values if len(magnitude) > 2 else magnitude.values
            right = magnitude.iloc[1::2].values if len(magnitude) > 2 else magnitude.values
            if len(left) > 0 and len(right) > 0:
                asymmetry = float(abs(left.mean() - right.mean()) / max(magnitude.mean(), 0.001) * 100)
                asymmetry = float(np.clip(asymmetry, 0, 50))
            else:
                asymmetry = 0.0

            # Fatigue index
            fi_values = self._compute_fatigue_index(magnitude, n_segments=5)
            fatigue_index = float(fi_values[-1]) if len(fi_values) > 0 else 0.5

            # Velocidad GPS
            velocity_avg = 0.0
            distance_km = 0.0
            if loc_df is not None and "velocity" in loc_df.columns:
                vel = pd.to_numeric(loc_df["velocity"], errors="coerce").dropna()
                if len(vel) > 0:
                    velocity_avg = float(vel.mean())
                    distance_km = float(velocity_avg * duration / 1000.0)

            # Running economy (pace en min/km)
            running_economy = 0.0
            if velocity_avg > 0.1:
                running_economy = round(1000.0 / (velocity_avg * 60.0), 1)

            # KLI - Knee Load Index
            kli_result = self._compute_kli(
                ground_shock=ground_shock,
                n_steps=n_steps,
                duration=duration,
                cadence=cadence_avg,
                asymmetry=asymmetry,
                fi_values=fi_values
            )

            # Cadencia temporal
            t_cad, cad_arr = self.processor.compute_cadence_over_time(df, peaks)

            return {
                "duration_minutes": round(duration_min, 2),
                "steps": n_steps,
                "cadence_avg": round(cadence_avg, 1),
                "ground_shock_avg": round(ground_shock, 3),
                "asymmetry": round(asymmetry, 2),
                "fatigue_index": round(fatigue_index, 3),
                "velocity_avg": round(velocity_avg, 3),
                "distance_km": round(distance_km, 3),
                "running_economy": running_economy,
                "fi_values": [round(float(v), 3) for v in fi_values],
                "cadence_over_time": {
                    "time": t_cad.tolist() if len(t_cad) > 0 else [],
                    "cadence": cad_arr.tolist() if len(cad_arr) > 0 else []
                },
                **kli_result
            }
        except Exception as e:
            return {"error": str(e), "kli": 0, "kli_status": "ERROR"}

    def _compute_fatigue_index(self, magnitude: pd.Series, n_segments: int = 5) -> np.ndarray:
        try:
            n = len(magnitude)
            if n < n_segments * 2:
                return np.array([0.5])
            seg_size = n // n_segments
            fi_values = []
            base = magnitude.iloc[:seg_size].mean()
            for i in range(n_segments):
                seg = magnitude.iloc[i * seg_size:(i + 1) * seg_size]
                fi = float(seg.mean() / max(base, 0.001))
                fi_values.append(np.clip(fi, 0.1, 2.0))
            return np.array(fi_values)
        except Exception:
            return np.array([0.5])

    def _compute_kli(self, ground_shock: float, n_steps: int, duration: float,
                     cadence: float, asymmetry: float, fi_values: np.ndarray) -> dict:
        try:
            # Carga base: impacto × pasos
            base_load = ground_shock * n_steps

            # Tasa de carga por minuto
            duration_min = max(duration / 60.0, 0.001)
            load_rate = base_load / duration_min

            # Penalización por cadencia baja (< 160 spm es más carga en rodilla)
            cadence_penalty = max(0, (160 - cadence) / 160) * 0.3 if cadence > 0 else 0

            # Penalización por asimetría
            asymmetry_penalty = (asymmetry / 50.0) * 0.2

            # Pendiente de fatiga
            if len(fi_values) >= 2:
                fatigue_slope = float(np.polyfit(range(len(fi_values)), fi_values, 1)[0])
            else:
                fatigue_slope = 0.0

            # KLI final (normalizado 0-100)
            kli_raw = (
                (ground_shock * n_steps / max(duration_min, 1)) * 0.5 +
                cadence_penalty * 10 +
                asymmetry_penalty * 10 +
                max(fatigue_slope, 0) * 20
            )
            kli = float(np.clip(kli_raw, 0, 100))

            # Estado
            if kli < 20:
                kli_status = "OK"
            elif kli < 40:
                kli_status = "WARNING"
            elif kli < 60:
                kli_status = "HIGH"
            else:
                kli_status = "CRITICAL"

            return {
                "kli": round(kli, 2),
                "kli_status": kli_status,
                "cumulative_load": round(base_load, 2),
                "load_per_step": round(base_load / max(n_steps, 1), 3),
                "load_rate": round(load_rate, 3),
                "fatigue_slope": round(fatigue_slope, 4),
            }
        except Exception:
            return {"kli": 0, "kli_status": "ERROR", "cumulative_load": 0,
                    "load_per_step": 0, "load_rate": 0, "fatigue_slope": 0}
