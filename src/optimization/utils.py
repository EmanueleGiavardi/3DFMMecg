import numpy as np
import matplotlib.pyplot as plt

def calculate_cos_phi(alpha, beta, omega, time_points):
    """
    Returns the matrix of base waveforms given alpha, beta, and temporal omega.
    Maintained for external utility compatibility.
    """
    if np.isscalar(alpha):
        alpha, beta, omega = [alpha], [beta], [omega]
        
    n_comp = len(alpha)
    n_obs = len(time_points)
    
    cos_phi = np.zeros((n_obs, n_comp))
    for i in range(n_comp):
        if not np.isnan(alpha[i]):
            al, be, om = alpha[i], beta[i], omega[i]
            phase = be + 2 * np.arctan(om * np.tan((time_points - al) / 2))
            cos_phi[:, i] = np.cos(phase)
    return cos_phi


def plot_ecg_beat(start_signal, fitted_wave, residual_signal, title="ECG Beat Fit", x_axis=None, y_lim=None):
    """
    Plots the starting signal (black), the fitted wave (red), and the new residual (blue).
    """
    plt.figure(figsize=(10, 6))
    x = x_axis if x_axis is not None else np.arange(len(start_signal))
    
    plt.plot(x, start_signal, label="Starting Signal", color='black', alpha=0.6, linewidth=1.5)
    plt.plot(x, fitted_wave, label="Fitted Wave", color='red', linewidth=2)
    plt.plot(x, residual_signal, label="New Residual", color='blue', alpha=0.8, linestyle='--')
    
    plt.title(title)
    plt.xlabel("Phase (rad)" if x_axis is not None else "Samples")
    plt.ylabel("Amplitude")
    
    if y_lim is not None:
        plt.ylim(y_lim)
        
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.show()