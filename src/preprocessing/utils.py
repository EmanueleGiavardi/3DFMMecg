import numpy as np
import scipy.signal

def detect_rpeaks(signal_arr, sRate, lowcut=1e-9, highcut=15, filter_order=1, 
                  integration_window=15, refractory=200, limit_factor=3, return_index=False):
    """
    Detects R-peaks in an ECG signal using the Pan-Tompkins algorithm.
    Implementation matching the original working version.
    """
    b, a = scipy.signal.butter(filter_order, [lowcut, highcut], btype='bandpass', fs=sRate)
    padded_signal = np.concatenate([np.repeat(signal_arr[0], sRate), signal_arr, np.repeat(signal_arr[-1], sRate)])
    signal_filt = scipy.signal.filtfilt(b, a, padded_signal, padlen=0)
    signal_filt = signal_filt[sRate : -sRate]
    
    signal_diff = np.diff(signal_filt)
    signal_squared = signal_diff**2
    # Keep length consistent after diff
    signal_squared = np.concatenate([[signal_squared[0]], signal_squared])
    
    # Moving integration window
    xc = scipy.signal.convolve(signal_squared, np.ones(integration_window), mode='full')
    difflen = len(xc) - len(signal_squared)
    start_idx = int(difflen / 2)
    end_idx = len(xc) - int(difflen / 2)
    xc = xc[start_idx:end_idx]
    signal_conv = xc
    
    peaks = [0]
    limit = np.mean(signal_conv) * limit_factor
    refractory_samples = int(sRate * (refractory / 1000.0))
    
    x = signal_conv
    for i in range(1, len(x) - 1):
        if (x[i] > limit) and (x[i] > x[i - 1]) and (x[i] > x[i + 1]) and ((i - peaks[-1]) > refractory_samples):
            peaks.append(i)
            
    peaks = peaks[1:]
    if return_index:
        return np.array(peaks)
    # Return 1-indexed seconds to match R logic (compensated in LeadProcessor)
    return (np.array(peaks) + 1) / sRate

def classify_rr_intervals(v_annot_n, freq):
    """
    Classifies RR intervals as subsequent, previous, or normal using heuristics.
    Implementation matching the original working version.
    """
    if len(v_annot_n) > 1:
        l_rr = np.diff(v_annot_n)
        # Filter outliers for median calculation
        if np.sum(l_rr > freq) > 0.75 * len(l_rr) and len(l_rr) > 3:
            l_rr_sorted = np.sort(l_rr)
            l_rr = l_rr_sorted[:-3]
        elif np.sum(l_rr > freq) > 0 and len(l_rr) > 3:
            l_rr = l_rr[l_rr <= freq]
        elif np.sum(l_rr > freq) > 0 and len(l_rr) == 3:
            l_rr = np.sort(l_rr)[:-1]

        M = np.median(l_rr) if len(l_rr) > 0 else np.nan
        fin = []
        for i in range(len(v_annot_n)):
            id_suc = [0, 0, 0]
            if i == 0:
                id_suc = [0, 0, 1] if (v_annot_n[1] - v_annot_n[0]) < 1.5 * M else [0, 1, 0]
            elif i == len(v_annot_n) - 1:
                id_suc = [1, 0, 0] if (v_annot_n[-1] - v_annot_n[-2]) < 1.5 * M else [0, 1, 0]
            else:
                rr_l = v_annot_n[i] - v_annot_n[i-1]
                rr_d = v_annot_n[i+1] - v_annot_n[i]
                if rr_l < 1.5 * M and rr_d < 1.5 * M: id_suc = [1, 0, 1]
                elif rr_l < 1.5 * M and rr_d >= 1.5 * M: id_suc = [1, 0, 0]
                elif rr_l >= 1.5 * M and rr_d < 1.5 * M: id_suc = [0, 0, 1]
                elif rr_l >= 1.5 * M and rr_d >= 1.5 * M: id_suc = [0, 1, 0]
            fin.append(id_suc)
        fin = np.array(fin)
    else:
        M = np.nan
        fin = np.array([])
    return fin, M

def compute_segment_ecg_beats(all_ecg, all_rpeaks, freq):
    """
    Extracts ECG segments corresponding to each beat.
    Implementation matching the original working version with floor/ceil logic.
    """
    fin, med_rr = classify_rr_intervals(all_rpeaks, freq)
    if not np.isnan(med_rr) and len(fin) > 0:
        datpac_len = len(all_ecg)
        bound_beats = np.zeros((len(all_rpeaks), 2), dtype=int)
        ya = False
        for j in range(len(all_rpeaks)):
            suc_j = fin[j]
            if np.array_equal(suc_j, [0, 0, 1]):
                rr_beat = all_rpeaks[j+1] - all_rpeaks[j]
                if not np.array_equal(fin[j+1], [0, 1, 0]):
                    bound_beats[j] = [0 if j == 0 and np.floor(all_rpeaks[j]-0.4*rr_beat) < 0 else np.floor(all_rpeaks[j]-0.4*rr_beat), np.ceil(all_rpeaks[j]+0.6*rr_beat)]
                else:
                    bound_beats[j] = [np.floor(all_rpeaks[j]-0.4*med_rr), np.ceil(all_rpeaks[j]+0.6*med_rr)]
            elif np.array_equal(suc_j, [1, 0, 0]) or (np.array_equal(suc_j, [0, 1, 0]) and j == len(all_rpeaks)-1):
                rr_beat = all_rpeaks[j] - all_rpeaks[j-1]
                if not np.array_equal(fin[j-1], [0, 1, 0]) and not np.array_equal(suc_j, [0, 1, 0]):
                    if j == len(all_rpeaks)-1 and np.ceil(all_rpeaks[j]+0.6*rr_beat) > datpac_len - 1:
                        bound_beats[j] = [np.floor(all_rpeaks[j]-0.4*rr_beat), datpac_len-1]
                        ya = True
                    else:
                        bound_beats[j] = [np.floor(all_rpeaks[j]-0.4*rr_beat), np.ceil(all_rpeaks[j]+0.6*rr_beat)]
                else:
                    bound_beats[j] = [np.floor(all_rpeaks[j]-0.4*med_rr), np.ceil(all_rpeaks[j]+0.6*med_rr)]
            elif np.array_equal(suc_j, [1, 0, 1]):
                bound_beats[j] = [np.floor(all_rpeaks[j]-0.4*(all_rpeaks[j]-all_rpeaks[j-1])), np.ceil(all_rpeaks[j]+0.6*(all_rpeaks[j+1]-all_rpeaks[j]))]
            elif np.array_equal(suc_j, [0, 1, 0]) and not ya:
                bound_beats[j] = [np.floor(all_rpeaks[j]-0.4*med_rr), np.ceil(all_rpeaks[j]+0.6*med_rr)]
        
        # Ensure continuity
        for j in range(1, len(bound_beats)):
            if bound_beats[j, 0] <= bound_beats[j-1, 1]: 
                bound_beats[j, 0] = bound_beats[j-1, 1] + 1
    else: 
        bound_beats = np.array([])
    return bound_beats

def compute_global_median_peaks(all_peaks_data, n_obs_signal, freq):
    """
    Computes global median peaks of an ECG and merges the peaks of multiple leads 
    into a single global signal.
    Implementation matching the original working version.
    all_peaks_data: list of (peak_val, lead_idx)
    """
    if not all_peaks_data:
        return np.array([]), np.array([]), np.array([])
        
    # In the original version, all_peaks_data is sorted by peak value
    aux_beats = np.array([p for p, l in all_peaks_data])
    aux_leads = np.array([l for p, l in all_peaks_data])
    
    v_beat, v_lead = [aux_beats[0]], [aux_leads[0]]
    beats, leads = [], []
    limit1, limit2 = int(103 * freq / 1000), int(322 * freq / 1000)
    
    j = 1
    while j < len(aux_beats):
        # If distance between current peak and previous is < 103ms, same heart beat
        if aux_beats[j] > (aux_beats[j-1] + limit1):
            # If we have more than 3 annotations (detected in at least 4 leads), consider it a beat
            if len(v_beat) > 3:
                beats.append(np.ceil(np.median(v_beat)))
                leads.append(list(v_lead))
            v_beat, v_lead = [aux_beats[j]], [aux_leads[j]]
        else:
            v_beat.append(aux_beats[j])
            v_lead.append(aux_leads[j])
            if j == len(aux_beats) - 1 and len(v_beat) > 3:
                beats.append(np.ceil(np.median(v_beat)))
                leads.append(list(v_lead))
        j += 1
    
    if not beats:
        return np.array([]), np.array([]), np.array([])
        
    final_beats, final_leads = [beats[0]], [leads[0]]
    ii = 1
    while ii < len(beats):
        if beats[ii] > (beats[ii-1] + limit2):
            final_beats.append(beats[ii])
            final_leads.append(leads[ii])
        ii += 1

    annos_pac = np.array(final_beats)
    leads_pac = final_leads
    seg_pac = compute_segment_ecg_beats(np.zeros(n_obs_signal), annos_pac, freq)
    
    return annos_pac, leads_pac, seg_pac
