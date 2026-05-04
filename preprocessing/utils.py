import numpy as np
import scipy.signal

def detect_rpeaks(signal_arr, sRate, lowcut=1e-9, highcut=15, filter_order=1, 
                  integration_window=15, refractory=200, limit_factor=3, return_index=False):
    """
    Detects R-peaks in an ECG signal using the Pan-Tompkins algorithm.
    """
    b, a = scipy.signal.butter(filter_order, [lowcut, highcut], btype='bandpass', fs=sRate)
    padded_signal = np.concatenate([np.repeat(signal_arr[0], sRate), signal_arr, np.repeat(signal_arr[-1], sRate)])
    signal_filt = scipy.signal.filtfilt(b, a, padded_signal, padlen=0)
    signal_filt = signal_filt[sRate:-sRate]
    
    # Derivative
    signal_diff = np.diff(signal_filt)
    # Squaring
    signal_sq = signal_diff ** 2
    # Moving average
    window_size = int(integration_window * sRate / 1000)
    signal_ma = np.convolve(signal_sq, np.ones(window_size)/window_size, mode='same')
    
    # Thresholding
    limit = limit_factor * np.median(signal_ma)
    peaks, _ = scipy.signal.find_peaks(signal_ma, height=limit, distance=int(refractory * sRate / 1000))
    
    if return_index:
        return peaks
    return peaks / sRate

def classify_rr_intervals(v_annot_n, freq):
    """
    Classifies RR intervals as subsequent, previous, or normal using heuristics (median-based threshold).
    """
    if len(v_annot_n) > 1:
        l_rr = np.diff(v_annot_n)
        med_rr = np.median(l_rr)
        
        # 1: Normal, 2: Short (likely artifact), 3: Long
        # Logic adapted from original R port
        fin = []
        for rr in l_rr:
            if rr < 0.85 * med_rr:
                fin.append(2)
            elif rr > 1.15 * med_rr:
                fin.append(3)
            else:
                fin.append(1)
        return np.array(fin), med_rr
    return np.array([]), np.nan

def compute_segment_ecg_beats(all_ecg, all_rpeaks, freq):
    """
    Extracts ECG segments corresponding to each beat.
    """
    fin, med_rr = classify_rr_intervals(all_rpeaks, freq)
    if not np.isnan(med_rr) and len(fin) > 0:
        # Simplified segment calculation for port parity
        # (Start and end of each beat segment)
        segments = []
        for i in range(len(all_rpeaks)):
            start = all_rpeaks[i] - 0.4 * med_rr
            end = all_rpeaks[i] + 0.6 * med_rr
            segments.append([max(0, start), min(len(all_ecg), end)])
        return np.array(segments)
    return np.array([])

def compute_global_median_peaks(all_peaks_data, n_obs_signal, freq):
    """
    Computes global median peaks of an ECG and merges the peaks of multiple leads 
    into a single global signal.
    Accepts a list of tuples: (peak_value, lead_idx) sorted by peak_value.
    """
    if not all_peaks_data:
        return np.array([]), np.array([]), np.array([])
        
    # Grouping peaks that are close to each other (within 100ms)
    groups = []
    current_group = [all_peaks_data[0]]
    
    for i in range(1, len(all_peaks_data)):
        peak_val, lead_idx = all_peaks_data[i]
        if (peak_val - current_group[-1][0]) < (0.1 * freq):
            current_group.append((peak_val, lead_idx))
        else:
            groups.append(current_group)
            current_group = [(peak_val, lead_idx)]
    groups.append(current_group)
        
    annos_pac = []
    leads_pac = []
    for g in groups:
        # The consensus peak is the median of all peaks in the group
        annos_pac.append(np.median([p for p, l in g]))
        # Keep track of which leads contributed to this consensus peak
        leads_pac.append([l for p, l in g])
        
    annos_pac = np.array(annos_pac)
    # Segments are calculated based on the consensus peaks
    seg_pac = compute_segment_ecg_beats(np.zeros(n_obs_signal), annos_pac, freq)
    
    return annos_pac, leads_pac, seg_pac
