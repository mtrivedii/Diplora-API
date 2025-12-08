import numpy as np
import math
import scipy.signal as sp_signal
from scipy.signal import butter, sosfiltfilt
import gc
import os
import filter as ecg_filter

class Signal:
    def __init__(self, hea_features):
        self.fs0        = float(hea_features['freq'])
        self.n_samples0 = int(hea_features['n_samples'])
        self.diag       = hea_features['diagnosis']

    @staticmethod
    def _resample_np(x, fs_in, fs_out):
        if abs(fs_in - fs_out) < 1e-9:
            return np.asarray(x, dtype=np.float32, order='C'), float(fs_out)
        g   = math.gcd(int(fs_out), int(fs_in))
        up  = int(fs_out // g)
        dn  = int(fs_in  // g)
        y = sp_signal.resample_poly(x, up, dn, axis=1)
        return y.astype(np.float32, copy=False), float(fs_out)

    @staticmethod
    def _bp_filter_np(x, fs, lowcut=0.5, highcut=40.0):
        nyq = 0.5 * fs
        low, high = lowcut/nyq, highcut/nyq
        sos = butter(4, [low, high], btype='band', output='sos')
        y = sosfiltfilt(sos, x, axis=1)
        return y.astype(np.float32, copy=False)

    @staticmethod
    def _apply_filter_np(x, fs, lowcut=0.5, highcut=40.0):
        """
        Apply selected denoising/filtering per channel.
        Mode selected via env FILTER_MODE: 'bandpass' (default), 'swt', 'ekf'.
        """
        mode = os.getenv("FILTER_MODE", "bandpass").strip().lower()
        if mode in ("bandpass", "bp", "bpf"):
            return Signal._bp_filter_np(x, fs, lowcut=lowcut, highcut=highcut)
        elif mode in ("swt", "modwt", "swt_modwt"):
            # Wavelet params via env, with safe defaults
            wavelet = os.getenv("FILTER_WAVELET", "sym4")
            level_env = os.getenv("FILTER_LEVEL", "")
            level = int(level_env) if level_env.isdigit() else None
            try:
                thr_scale = float(os.getenv("FILTER_THR_SCALE", "1.0"))
            except Exception:
                thr_scale = 1.0
            out = np.empty_like(x, dtype=np.float32)
            for c in range(x.shape[0]):
                out[c] = ecg_filter.swt_modwt_denoise(x[c], wavelet=wavelet, level=level, thr_scale=thr_scale)
            return out
        elif mode in ("ekf", "kalman", "ekf_denoise"):
            # EKF params via env
            def _getf(name, default):
                try:
                    return float(os.getenv(name, str(default)))
                except Exception:
                    return float(default)
            q1 = _getf("EKF_Q1", 1.5e-4)
            q2 = _getf("EKF_Q2", 2e-3)
            q3 = _getf("EKF_Q3", 8e-4)
            r  = _getf("EKF_R",  0.02)
            out = np.empty_like(x, dtype=np.float32)
            for c in range(x.shape[0]):
                out[c] = ecg_filter.ekf_denoise(x[c], q1=q1, q2=q2, q3=q3, r=r)
            return out
        else:
            # Fallback to bandpass
            return Signal._bp_filter_np(x, fs, lowcut=lowcut, highcut=highcut)

    @staticmethod
    def _normalize_np(x):
        m = x.mean(axis=1, keepdims=True)
        s = x.std(axis=1, keepdims=True)
        return ((x - m) / (s + 1e-8)).astype(np.float32, copy=False)

    @staticmethod
    def _interp_to_len(x, out_len):
        """Linear time interpolation per channel to exactly out_len samples."""
        C, T = x.shape
        if T == out_len: return x
        # src grid [0..T-1] -> dst grid [0..out_len-1]
        src = np.linspace(0.0, 1.0, num=T, endpoint=True, dtype=np.float64)
        dst = np.linspace(0.0, 1.0, num=out_len, endpoint=True, dtype=np.float64)
        out = np.empty((C, out_len), dtype=np.float32)
        for c in range(C):
            out[c] = np.interp(dst, src, x[c].astype(np.float64, copy=False)).astype(np.float32, copy=False)
        return out

    def preprocess_window(self, raw, *, target_fs=500, lowcut=0.5, highcut=40.0,
                          win_len=5000, overlap=0, start=None, window_index=None,
                          pad_mode='interp'):
        """
        Returns one window [C, win_len] covering the requested region.
        pad_mode: 'interp' -> interpolate short recordings to win_len
                  'zero'   -> zero-pad short or tail
        """
        x = np.asarray(raw, dtype=np.float32, order='C')     # [C,T0]
        x, fs = self._resample_np(x, self.fs0, float(target_fs))
        x = self._apply_filter_np(x, fs, lowcut, highcut)
        x = self._normalize_np(x)

        C, T = x.shape
        step = int(win_len - overlap) if win_len > overlap else 1

        # compute start position
        if start is None:
            k = int(window_index or 0)
            start = k * step

        # case: whole recording shorter than window -> interpolate (or pad)
        if T <= win_len:
            if pad_mode == 'interp':
                out = self._interp_to_len(x, win_len)      # covers all data by stretching
            else:
                out = np.zeros((C, win_len), dtype=np.float32); out[:, :T] = x
            del x; gc.collect()
            return out, fs

        # clamp start for last window if slightly beyond due to rounding
        if start > T - win_len:
            start = max(0, T - win_len)

        end = start + win_len
        if end <= T:
            out = x[:, start:end]
        else:
            # tail window overshoots: zero-pad remainder
            out = np.zeros((C, win_len), dtype=np.float32)
            n_avail = max(0, T - start)
            if n_avail > 0:
                out[:, :n_avail] = x[:, start:start+n_avail]
        del x; gc.collect()
        return out, fs

    @staticmethod
    def estimate_windows_and_meta(hea_features, target_fs=500, win_len=5000, overlap=0, cover_all=True):
        fs0 = float(hea_features['freq'])
        n0  = int(hea_features['n_samples'])
        n_resamp = int(round(n0 * (float(target_fs)/fs0)))
        step = int(win_len - overlap) if win_len > overlap else 1
        if n_resamp <= win_len:
            nwin = 1
        else:
            if cover_all:
                # ceil coverage: include a final window anchored at the end
                nwin = 1 + math.ceil(max(0, n_resamp - win_len) / step)
            else:
                # floor (only full steps)
                nwin = 1 + max(0, (n_resamp - win_len) // step)
        return n_resamp, nwin, step

    def preprocess_full(self, raw, *, target_fs=500, lowcut=0.5, highcut=40.0):
        """
        Preprocess the entire recording once: resample, bandpass filter, and normalize.
        Returns array [C, T_resampled] and the effective sampling frequency.
        """
        x = np.asarray(raw, dtype=np.float32, order='C')
        x, fs = self._resample_np(x, self.fs0, float(target_fs))
        x = self._apply_filter_np(x, fs, lowcut, highcut)
        x = self._normalize_np(x)
        return x.astype(np.float32, copy=False), fs


# ============================================================================
# R-peak detection and HRV feature extraction
# ============================================================================

def detect_r_peaks(signal, fs):
    """
    Pan-Tompkins-like R-peak detection for ECG signals.
    
    Args:
        signal: 1D numpy array of ECG signal (single lead, preferably lead II)
        fs: sampling frequency in Hz
    
    Returns:
        r_peaks: numpy array of R-peak indices
    """
    # Step 1: Bandpass filter (5-15 Hz for QRS complex)
    nyq = 0.5 * fs
    low, high = 5.0 / nyq, 15.0 / nyq
    # Clamp to valid range
    low = max(0.01, min(low, 0.99))
    high = max(low + 0.01, min(high, 0.99))
    
    try:
        sos = butter(2, [low, high], btype='band', output='sos')
        filtered = sosfiltfilt(sos, signal)
    except Exception:
        # Fallback if filter fails
        filtered = signal.copy()
    
    # Step 2: Differentiation (emphasize slope)
    diff = np.diff(filtered)
    diff = np.pad(diff, (0, 1), mode='edge')  # Keep same length
    
    # Step 3: Squaring (make all values positive and amplify)
    squared = diff ** 2
    
    # Step 4: Moving window integration (~150ms window)
    window_size = int(0.15 * fs)  # 150ms
    window_size = max(1, window_size)
    integrated = np.convolve(squared, np.ones(window_size) / window_size, mode='same')
    
    # Step 5: Adaptive thresholding
    # Find peaks in integrated signal
    # Use a minimum distance of ~200ms (typical RR interval minimum)
    min_distance = int(0.2 * fs)
    
    # Simple peak detection: local maxima above threshold
    threshold = np.mean(integrated) + 0.5 * np.std(integrated)
    
    peaks = []
    for i in range(1, len(integrated) - 1):
        if integrated[i] > threshold and integrated[i] > integrated[i-1] and integrated[i] > integrated[i+1]:
            # Check minimum distance from last peak
            if len(peaks) == 0 or (i - peaks[-1]) >= min_distance:
                peaks.append(i)
    
    # Refine peaks: find actual R-peak in original signal near detected peaks
    # Look in ±50ms window around detected peak
    search_window = int(0.05 * fs)
    refined_peaks = []
    for peak_idx in peaks:
        start = max(0, peak_idx - search_window)
        end = min(len(signal), peak_idx + search_window)
        if start < end:
            local_max_idx = start + np.argmax(np.abs(signal[start:end]))
            refined_peaks.append(local_max_idx)
    
    return np.array(refined_peaks, dtype=np.int32)


def compute_hrv_features(r_peaks, fs, n_intervals=10):
    """
    Compute HRV features from R-peak locations.
    
    Args:
        r_peaks: numpy array of R-peak indices
        fs: sampling frequency in Hz
        n_intervals: number of RR intervals to extract (default 10)
    
    Returns:
        features: numpy array of shape [12] containing:
                  [RR_1, ..., RR_10, RMSSD, SDNN]
                  RR intervals in milliseconds
    """
    features = np.zeros(12, dtype=np.float32)
    
    # Need at least 2 peaks to compute RR intervals
    if len(r_peaks) < 2:
        # Return zeros (will be handled as missing data)
        return features
    
    # Compute RR intervals in milliseconds
    rr_intervals = np.diff(r_peaks) / fs * 1000.0  # Convert to ms
    
    # Take last n_intervals (most recent rhythm)
    if len(rr_intervals) >= n_intervals:
        last_rr = rr_intervals[-n_intervals:]
    else:
        # Pad with mean if not enough intervals
        last_rr = np.zeros(n_intervals, dtype=np.float32)
        last_rr[:len(rr_intervals)] = rr_intervals
        if len(rr_intervals) > 0:
            last_rr[len(rr_intervals):] = np.mean(rr_intervals)
    
    # Store last 10 RR intervals
    features[:10] = last_rr
    
    # Compute RMSSD (root mean square of successive differences)
    if len(rr_intervals) >= 2:
        successive_diffs = np.diff(rr_intervals)
        rmssd = np.sqrt(np.mean(successive_diffs ** 2))
        features[10] = rmssd
    
    # Compute SDNN (standard deviation of RR intervals)
    if len(rr_intervals) >= 2:
        sdnn = np.std(rr_intervals)
        features[11] = sdnn
    
    return features