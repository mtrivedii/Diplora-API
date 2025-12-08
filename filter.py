# IMPORT
import numpy as np
import pywt
from scipy.signal import butter, filtfilt, sosfiltfilt
import numpy as np
from filterpy.kalman import ExtendedKalmanFilter


## FILTER USING MAXIMAL OVERLAP DISCRETE WAVELET TRANSFORM (MODWT)
def swt_modwt_denoise(x, wavelet="sym4", level=None, thr_scale=1.0, return_coeffs=False):
    x = np.asarray(x, float)
    n = len(x)
    max_level = pywt.swt_max_level(n)
    if level is None or level > max_level:
        level = max_level if max_level > 0 else 1
    coeffs = pywt.swt(x, wavelet=wavelet, level=level)
    den_coeffs = []
    for (cA, cD) in coeffs:
        sigma = 1.4826 * np.median(np.abs(cD - np.median(cD))) + 1e-12
        thr = thr_scale * sigma * np.sqrt(2.0*np.log(n))
        cD_d = pywt.threshold(cD, thr, mode="soft")
        den_coeffs.append((cA, cD_d))
    x_den = pywt.iswt(den_coeffs, wavelet=wavelet)
    return (x_den, den_coeffs) if return_coeffs else x_den

# --- High-pass filter to reduce baseline wander ---
def high_pass_filter(x, cutoff=1, fs=500, order=4):
    nyq = 0.5 * fs
    sos = butter(order, cutoff/nyq, btype='high', output='sos')
    return sosfiltfilt(sos, x)

## FILTER USING EXTENDED KALMAN FILTER (EKF)
def ekf_denoise(signal, q1=1.5e-4, q2=2e-3, q3=8e-4, r=0.02):
    """
    Denoise an ECG signal using a robust EKF with baseline tracking.
    Input:  signal (1D numpy array)
    Output: denoised signal (1D numpy array)
    """

    x_in = np.asarray(signal, float).ravel()
    fs, dt = 500, 1/500
    x_hp = high_pass_filter(x_in)

    # --- EKF model: [value, slope, baseline] ---
    F = np.array([[1.0, dt, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]])
    def HJacobian_at(_): return np.array([[1.0, 0.0, 1.0]])
    def hx(state): return np.array([state[0] + state[2]])

    ekf = ExtendedKalmanFilter(dim_x=3, dim_z=1)
    ekf.x = np.array([x_hp[0], 0.0, 0.0])
    ekf.P = np.diag([1.0, 1.0, 1e3])
    ekf.F = F
    ekf.Q = np.diag([q1, q2, q3])
    ekf.R = np.array([[r]])

    out = np.zeros_like(x_hp)
    for i, z in enumerate(x_hp):
        ekf.predict()
        ekf.update(z, HJacobian_at, hx)
        out[i] = ekf.x[0] + ekf.x[2]

    return out