import numpy as np
import scipy.signal
from skmisc.loess import loess

import numpy as np
import scipy.signal
from skmisc.loess import loess

# =======================================================================
# LOW-LEVEL FUNCTIONS
# =======================================================================

def detect_rpeaks(signal_arr, sRate, lowcut=1e-9, highcut=15, filter_order=1, 
                  integration_window=15, refractory=200, limit_factor=3, return_index=False):
    b, a = scipy.signal.butter(filter_order, [lowcut, highcut], btype='bandpass', fs=sRate)
    padded_signal = np.concatenate([np.repeat(signal_arr[0], sRate), signal_arr, np.repeat(signal_arr[-1], sRate)])
    signal_filt = scipy.signal.filtfilt(b, a, padded_signal, padlen=0)
    signal_filt = signal_filt[sRate : -sRate]
    signal_diff = np.diff(signal_filt)
    signal_squared = signal_diff**2
    signal_squared = np.concatenate([[signal_squared[0]], signal_squared])
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
    # R returns 1-indexed peaks / sRate
    return (np.array(peaks) + 1) / sRate

def give_sucesive_ecg(v_annot_n, freq):
    if len(v_annot_n) > 1:
        l_rr = np.diff(v_annot_n)
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

def give_segments_ecg(all_ecg, all_rpeaks, freq):
    fin, med_rr = give_sucesive_ecg(all_rpeaks, freq)
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
        for j in range(1, len(bound_beats)):
            if bound_beats[j, 0] <= bound_beats[j-1, 1]: bound_beats[j, 0] = bound_beats[j-1, 1] + 1
    else: bound_beats = np.array([])
    return bound_beats

def give_anno_med_app(anno_leads, n_obs_signal, freq):
    # aux_beates -> array di indici dei picchi R
    # aux_leads -> array inutile dei numeri delle derivazioni   
    aux_beats, aux_leads = anno_leads[0], anno_leads[1]
    v_beat, v_lead = [aux_beats[0]], [aux_leads[0]]
    beats, leads = [], []
    limit1, limit2 = int(103 * freq / 1000), int(322 * freq / 1000)
    
    j = 1
    while j < len(aux_beats):
        # se la distanza tra l'annotazione di picco corrente e quella precedente è minore della 
        # threshold (103 ms), siamo nello stesso battito cardiaco visto da derivazioni diverse

        # se la distanza è maggiore, siamo in un altro battito
        if aux_beats[j] > (aux_beats[j-1] + limit1):
            # se abbiamo più di 3 annotazioni di picco nello stesso battito, calcoliamo la mediana
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
        
    final_beats, final_leads = [beats[0]], [leads[0]]
    ii, k = 1, 1
    while ii < len(beats):
        if beats[ii] > (beats[ii-1] + limit2):
            final_beats.append(beats[ii])
            final_leads.append(leads[ii])
            k += 1
        ii += 1

    annos_pac = np.array(final_beats)
    leads_pac = final_leads
    seg_pac = give_segments_ecg(np.zeros(n_obs_signal), annos_pac, freq)
    if len(seg_pac) > 0:
        if seg_pac[0, 0] < 0: seg_pac[0, 0] = 0
        if seg_pac[-1, 1] > n_obs_signal - 1: seg_pac[-1, 1] = n_obs_signal - 1
    else: seg_pac = np.array([np.nan, np.nan])
    
    # annos_pac -> array di indici dei picchi R (globali)
    # leads_pac -> array di indici delle derivazioni
    # seg_pac -> array di indici dei segmenti ECG
    return annos_pac, leads_pac, seg_pac

# =======================================================================
# MID-LEVEL FUNCTIONS (Artifact Drops)
# =======================================================================

def drop_anno1_app(data, anno, rr, seg, freq):
    if len(anno) <= 1:
        return anno.tolist() if isinstance(anno, np.ndarray) else list(anno), list(range(len(anno)))
        
    anno_new = np.array(anno)
    rr_new = np.array(rr)
    seg_new = np.array(seg)
    anno_orig = np.array(anno)
    
    thr_d_new = (2/3) * np.median(rr_new)
    
    def get_dist_anno(a):
        return np.diff(a) if len(a) > 1 else np.array([])
        
    dist_anno_new = get_dist_anno(anno_new)
    
    def calc_seguir(rr_arr, seg_arr, d_anno):
        if len(rr_arr) == 0: return False
        c1 = np.sum(rr_arr < thr_d_new) > 0
        c2 = np.sum(d_anno > (4/3)*np.median(rr_arr)) >= 1
        c3_part1 = len(seg_arr) > (4/3)*(len(data)/freq)
        if len(rr_arr) > 1 and np.mean(rr_arr) != 0:
            std_rr = np.std(rr_arr, ddof=1)
            c3_part2 = (std_rr / np.mean(rr_arr)) > 0.2
        else:
            c3_part2 = False
        return c1 or c2 or (c3_part1 and c3_part2)

    seguir = calc_seguir(rr_new, seg_new, dist_anno_new)
    time = 1
    
    while seguir:
        aqui = False
        cond = rr_new < thr_d_new
        
        c2 = np.sum(dist_anno_new > (4/3)*np.median(rr_new)) >= 1
        c1 = np.sum(rr_new < thr_d_new) == 0
        if c2 and c1 and time == 1:
            cond = dist_anno_new > (4/3)*np.median(rr_new)
            time = 2
            aqui = True
        else:
            c3_part1 = len(seg_new) > (4/3)*(len(data)/freq)
            if len(rr_new) > 1 and np.mean(rr_new) != 0:
                c3_part2 = (np.std(rr_new, ddof=1) / np.mean(rr_new)) > 0.2
            else:
                c3_part2 = False
            if c3_part1 and c3_part2:
                cond = np.array([True]*(len(seg_new)-2) + [False])
                
        cond_idx = np.where(cond)[0]
        if len(cond_idx) == 0: break
            
        ext_mismo1 = anno_new[cond_idx] - int(np.ceil(0.02*freq))
        ext_mismo2 = anno_new[cond_idx] + int(np.ceil(0.02*freq))
        ext_mas_uno1 = anno_new[cond_idx+1] - int(np.ceil(0.02*freq))
        ext_mas_uno2 = anno_new[cond_idx+1] + int(np.ceil(0.02*freq))
        ext_menos_uno1 = anno_new[cond_idx] - int(np.ceil(0.02*freq))
        ext_menos_uno2 = anno_new[cond_idx] + int(np.ceil(0.02*freq))
        
        def get_range_amp(d, start, end):
            s = max(0, int(start))
            e = min(len(d), int(end)+1)
            if s >= e: return 0
            return np.max(d[s:e]) - np.min(d[s:e])

        mismo = np.array([get_range_amp(data, ext_mismo1[i], ext_mismo2[i]) for i in range(len(cond_idx))])
        mas_uno = np.array([get_range_amp(data, ext_mas_uno1[i], ext_mas_uno2[i]) for i in range(len(cond_idx))])
        
        une = []
        for i, j_idx in enumerate(cond_idx):
            s = max(0, int(seg_new[j_idx, 0]))
            e = min(len(data), int(seg_new[j_idx, 1])+1)
            dd = data[s:e] if s < e else np.array([0])
            dd_amp = np.max(dd) - np.min(dd)
            
            cond1 = abs(mismo[i] - mas_uno[i]) > 0.25 * dd_amp
            cond2 = dd_amp > 2 * abs(mismo[i]) and dd_amp > 2 * abs(mas_uno[i])
            cond3 = abs(mismo[i] - mas_uno[i]) < 0.25 * dd_amp
            
            if (cond1 or (cond2 and cond3)) and (seg_new[j_idx+1, 0] - seg_new[j_idx, 1] < (1/3)*np.median(rr_new)):
                if mismo[i] > mas_uno[i]:
                    une.append(j_idx + 1)
                else:
                    une.append(j_idx)
                    
        une = np.unique(une)
        anno_drop = anno_new.copy()
        if len(une) > 0:
            anno_drop = np.delete(anno_new, une)
            anno_new = anno_drop.copy()
            seg_new = np.delete(seg_new, une, axis=0)
            
        rr_new = get_dist_anno(anno_new)
        
        elimina = []
        if len(anno_drop) > 2 and time == 1:
            for j in range(1, len(anno_drop)-1):
                inf = anno_drop[j] - int(np.ceil(0.02*freq))
                sup = anno_drop[j] + int(np.ceil(0.02*freq))
                inf_sig = anno_drop[j+1] - int(np.ceil(0.02*freq))
                sup_sig = anno_drop[j+1] + int(np.ceil(0.02*freq))
                inf_ant = anno_drop[j-1] - int(np.ceil(0.02*freq))
                sup_ant = anno_drop[j-1] + int(np.ceil(0.02*freq))
                
                def get_max(d, start, end):
                    s = max(0, int(start))
                    e = min(len(d), int(end)+1)
                    if s >= e: return -np.inf
                    return np.max(d[s:e])
                    
                val_curr = get_max(data, inf, sup)
                val_ant = get_max(data, inf_ant, sup_ant)
                val_sig = get_max(data, inf_sig, sup_sig)
                
                cond_ant = (2 * val_ant < val_curr)
                cond_sup = (2 * val_sig < val_curr)
                
                s = max(0, int(seg_new[j, 0]))
                e = min(len(data), int(seg_new[j, 1]))
                len_seg = e - s
                
                std_rr = np.std(rr_new, ddof=1) if len(rr_new) > 1 else 0
                mean_rr = np.mean(rr_new) if len(rr_new) > 0 else 1
                if mean_rr == 0: mean_rr = 1
                
                if (cond_ant or cond_sup) and len_seg < (2/3)*freq and (std_rr / mean_rr) < 0.2:
                    if cond_ant: elimina.append(j-1)
                    else: elimina.append(j+1)
                        
            if len(elimina) > 0:
                elimina = np.unique(elimina)
                anno_drop = np.delete(anno_drop, elimina)
                seg_new = np.delete(seg_new, elimina, axis=0)
                
        time = 2
        
        if len(elimina) > 0:
            anno_new = anno_drop.copy()
            
        rr_new = get_dist_anno(anno_new)
        if len(rr_new) > 0:
            thr_d_new = (2/3)*np.median(rr_new)
        
        if len(anno_new) <= 1:
            seguir = False
        else:
            dist_anno_new = get_dist_anno(anno_new)
            seguir = np.sum(rr_new < thr_d_new) > 0
            if len(une) == 0 and len(elimina) == 0:
                seguir = False
                
    pos_anno_new = []
    for a in anno_new:
        pos_anno_new.append(np.where(anno_orig == a)[0][0])
        
    return anno_new.tolist(), pos_anno_new

def drop_anno2_app(data, anno, rr, seg, anno_orig, freq):
    anno_new = np.array(anno)
    rr_new = np.array(rr)
    seg_new = np.array(seg)
    anno_orig = np.array(anno_orig)
    
    if len(anno_new) <= 1:
        return anno_new.tolist(), list(range(len(anno_new)))
        
    primero = False
    segundo = False
    
    def get_a_b(seg_n, d, anno_n):
        if len(seg_n) == 0: return np.array([]), np.array([])
        s1 = max(0, int(seg_n[0, 0]))
        e1 = min(len(d), int(seg_n[0, 1]))
        if e1 > s1: e_a = max(s1 + int(np.ceil(0.02*(e1 - s1))), 1 + int(np.ceil(0.02*(e1 - s1))))
        else: e_a = s1
        a_arr = d[s1:e_a]
        
        s_end = max(0, int(seg_n[-1, 0]))
        e_end = min(len(d), int(seg_n[-1, 1]))
        if e_end > s_end:
            c = int(np.ceil(0.02*(e_end - s_end)))
            s_b = min(len(d) - c, e_end) - c
        else: s_b = e_end
        b_arr = d[max(0, s_b):e_end]
        return a_arr, b_arr

    a_arr, b_arr = get_a_b(seg_new, data, anno_new)
    
    ini1 = max(0, int(seg_new[0, 0]))
    e1 = min(len(data), int(seg_new[0, 1]))
    c_len = e1 - ini1
    ini2 = ini1 - 1 + int(np.ceil(0.15 * c_len))
    fin1 = e1 + 1 - int(np.ceil(0.15 * c_len))
    fin2 = e1
    
    data_norm = data[ini1:fin2]
    if len(data_norm) > 0:
        val_min = np.min(data_norm)
        val_max = np.max(data_norm)
        v_min = np.argmin(data_norm)
        v_max = np.argmax(data_norm)
        diff1 = abs(val_min - val_max)
        if v_min <= (ini2 - ini1) or v_max <= (ini2 - ini1):
            primero = True
    else:
        diff1 = 0

    diff_arr = []
    for j in range(1, len(seg_new)):
        i1 = max(0, int(seg_new[j, 0]))
        e2 = min(len(data), int(seg_new[j, 1]))
        dn = data[i1:e2]
        if len(dn) > 0: diff_arr.append(abs(np.min(dn) - np.max(dn)))
        else: diff_arr.append(0)
            
    if len(diff_arr) > 0:
        med_diff = np.median(diff_arr)
        if diff1 > (2/3)*med_diff or diff1 < (2/3)*med_diff:
            if primero: segundo = True
            
    def get_sd_mean(arr):
        if len(arr) < 2: return 0, 1
        return np.std(arr, ddof=1), np.mean(arr)
        
    def calc_seguir(sn, an, a_a, b_a, seg2):
        if len(sn) <= 1: return False
        med_seg = np.median(sn[:, 1] - sn[:, 0])
        c1 = (sn[0, 1] - sn[0, 0]) < (0.9 * med_seg) and len(an) > 1
        sd_a, mean_a = get_sd_mean(a_a)
        c2 = (sd_a / abs(mean_a)) < 0.01 if mean_a != 0 else False
        c3 = (sd_a / abs(mean_a)) < 0.02 and len(an) >= 0.04*freq if mean_a != 0 else False
        return c1 or c2 or c3 or seg2
        
    seguir = calc_seguir(seg_new, anno_new, a_arr, b_arr, segundo)
    
    while seguir:
        anno_drop = anno_new[1:]
        anno_new = anno_drop
        rr_new = np.diff(anno_new) if len(anno_new) > 1 else np.array([])
        seg_new = seg_new[1:]
        if len(anno_new) <= 1:
            seguir = False
        else:
            med_seg = np.median(seg_new[:, 1] - seg_new[:, 0])
            seguir = (seg_new[0, 1] - seg_new[0, 0]) < (0.85 * med_seg) and len(anno_new) > 1

    if len(anno_new) > 1:
        med_seg = np.median(seg_new[:, 1] - seg_new[:, 0])
        sd_b, mean_b = get_sd_mean(b_arr)
        c1 = (seg_new[-1, 1] - seg_new[-1, 0]) < (0.9 * med_seg)
        c2 = (sd_b / abs(mean_b)) < 0.01 if mean_b != 0 else False
        c3 = (sd_b / abs(mean_b)) < 0.02 and len(anno_new) >= 0.04*freq if mean_b != 0 else False
        seguir = c1 or c2 or c3
        
    while seguir:
        anno_drop = anno_new[:-1]
        anno_new = anno_drop
        rr_new = np.diff(anno_new) if len(anno_new) > 1 else np.array([])
        seg_new = seg_new[:-1]
        if len(anno_new) <= 1:
            seguir = False
        else:
            med_seg = np.median(seg_new[:, 1] - seg_new[:, 0])
            seguir = (seg_new[-1, 1] - seg_new[-1, 0]) < (0.85 * med_seg) and len(anno_new) > 1
            
    pos_anno_new = []
    for a in anno_new:
        idx = np.where(anno_orig == a)[0]
        if len(idx) > 0: pos_anno_new.append(idx[0])
        
    return anno_new.tolist(), pos_anno_new

def drop_anno3_app(anno, rr, seg, anno_orig):
    anno_new = np.array(anno)
    rr_new = np.array(rr)
    seg_new = np.array(seg)
    anno_orig = np.array(anno_orig)
    
    if len(anno_new) > 1 and len(seg_new) > 0:
        seg_len = seg_new[:, 1] - seg_new[:, 0]
        pos_in_seg = anno_new - seg_new[:, 0] + 1
        cond1 = (seg_len * 0.35) > pos_in_seg
        cond2 = pos_in_seg > (0.45 * seg_len)
        seguir = np.sum(cond1 | cond2) >= 1
        if seguir:
            donde = np.where(cond1 | cond2)[0]
            anno_new = np.delete(anno_new, donde)
            seg_new = np.delete(seg_new, donde, axis=0)
            rr_new = np.diff(anno_new) if len(anno_new) > 1 else np.array([])
            
    pos_anno_new = []
    for a in anno_new:
        idx = np.where(anno_orig == a)[0]
        if len(idx) > 0: pos_anno_new.append(idx[0])
        
    return anno_new.tolist(), pos_anno_new

def drop_anno5_app(data, anno, rr, seg, anno_orig):
    anno_new = np.array(anno)
    rr_new = np.array(rr)
    seg_new = np.array(seg)
    anno_orig = np.array(anno_orig)
    seguir = False
    
    if len(anno_new) > 1:
        dife = np.zeros(len(anno_new))
        amp = np.zeros(len(anno_new))
        condi_key = np.zeros(len(anno_new), dtype=bool)
        revisar_key = []
        
        for j in range(len(anno_new)):
            ini1 = max(0, int(seg_new[j, 0]))
            e2 = min(len(data), int(seg_new[j, 1]))
            c_len = e2 - ini1
            
            ini2 = ini1 - 1 + int(np.ceil(0.05 * c_len))
            fin1 = e2 + 1 - int(np.ceil(0.05 * c_len))
            fin2 = e2
            
            data_norm = data[ini1:fin2]
            
            if len(data_norm) > 0:
                s1 = 0
                e1 = max(0, ini2 - ini1 + 1)
                s2 = max(0, fin1 - ini1)
                e2_dn = max(0, fin2 - ini1)
                
                d_ini = data_norm[s1:e1]
                d_fin = data_norm[s2:e2_dn]
                
                val_ini = np.median(d_ini) if len(d_ini) > 0 else 0
                val_fin = np.median(d_fin) if len(d_fin) > 0 else 0
                
                dife[j] = abs(val_fin - val_ini)
                amp[j] = abs(np.max(data_norm) - np.min(data_norm))
                
                condi_key[j] = dife[j] > 0.15 * amp[j]
                if condi_key[j]: revisar_key.append(j)
            
        if np.sum(condi_key) >= 4:
            condi_key = np.ones(len(anno_new), dtype=bool)
            dife = np.full(len(anno_new), 999.0)
            amp = np.ones(len(anno_new))
            
        if len(revisar_key) > 0: seguir = True
        
        if seguir:
            donde = np.where(dife > amp)[0].tolist() + revisar_key
            donde = np.unique(donde)
            
            anno_new = np.delete(anno_new, donde)
            rr_new = np.diff(anno_new) if len(anno_new) > 1 else np.array([])
            seg_new = np.delete(seg_new, donde, axis=0)
            
    pos_anno_new = []
    for a in anno_new:
        idx = np.where(anno_orig == a)[0]
        if len(idx) > 0: pos_anno_new.append(idx[0])
        
    return anno_new.tolist(), pos_anno_new

def lead_pre_pan_tom_app(data, freq):
    data = np.nan_to_num(data, copy=True)
    n_samples = len(data)
    # check: almeno il 75% dei campioni non è zero
    # altrimenti, restituisce array vuoti per questa specifica derivazione
    if np.sum(data != 0) > 0.75 * n_samples:
        dd = data.copy()
        # RIMOZIONE BASELINE WANDER
        # TODO: provare un semplice filtro passalto per la rimozione di baseline wander
        # TODO: cercare anche presenza di power-line interference
        l = loess(np.arange(n_samples), dd, span=0.2, degree=2)
        l.fit()
        loess_fitted = l.outputs.fitted_values
        loess_data = dd - loess_fitted

        # TODO: stampare ECG prima e dopo l'applicazione del filtro LOESS 
        
        # RILEVAMENTO PICCHI R TRAMITE PAN-TOMPKINS
        try:
            peaks_sec = detect_rpeaks(loess_data, freq)
            # Reproduce R floating point noise, then subtract 1 for 0-indexing
            loess_rpeaks = (peaks_sec * freq) - 1
        except:
            loess_rpeaks = np.array([])

        # se ci sono meno di due (TODO: sostituire con una threshold) picchi     
        # gestisce i picchi anomali: picchi estremamente alti o estremamente bassi
        # vengono portati al valore di mediana del segnale

        # prima viene rimossa la baseline wander, poi vengono cercati i picchi, che 
        # vengono salvati in loess_rpeaks. Poi riesegue lo stesso fitro LOESS fino a che 
        # non vengono rilevati più di 2 picchi... bah
        # TODO: refactoring... più che altro capire se tutto questo blocco ha davvero senso
        while len(loess_rpeaks) <= 2:
            q1 = np.quantile(dd, 0.9999)
            if np.max(dd) >= q1: dd[dd >= q1] = np.median(dd)
            q2 = np.quantile(dd, 0.0001)
            if np.min(dd) <= q2: dd[dd <= q2] = np.median(dd)
                

            l = loess(np.arange(n_samples), dd, span=0.2, degree=2)
            l.fit()
            loess_fitted = l.outputs.fitted_values
            loess_data = dd - loess_fitted
            try:
                peaks_sec = detect_rpeaks(loess_data, freq)
                loess_rpeaks = (peaks_sec * freq) - 1
            except:
                break
                
        loess_seg = give_segments_ecg(loess_data, loess_rpeaks, freq)
        loess_rr = np.diff(loess_rpeaks) if len(loess_rpeaks) > 1 else []
        
        # matrice 3xN con:
        #  1234 (codice fittizio per il paziente?) | indice del picco | codice di stato del picco (es. 0 = ok)
        tab_codes = np.zeros((len(loess_rpeaks), 3), dtype=int)
        tab_codes[:, 0] = 1234
        tab_codes[:, 1] = np.arange(1, len(loess_rpeaks) + 1)
        tab_codes[:, 2] = 0
        
        # drop_anno1_app fa dei controlli (?), dopodichè vengono ricalcolati i segmenti e le distanze RR
        # TODO: valutare se tenere o meno
        paso1_beats, paso1_idx = drop_anno1_app(loess_data, loess_rpeaks, loess_rr, loess_seg, freq)
        seg1 = give_segments_ecg(loess_data, paso1_beats, freq)
        rr1 = np.diff(paso1_beats) if len(paso1_beats) > 1 else []
        
        if len(paso1_beats) > 0:
            loess_rpeaks_end = np.array(paso1_beats)
            loess_seg_end = np.array(seg1)
            loess_rr_end = np.array(rr1)
        else:
            loess_rpeaks_end = np.array([])
            loess_seg_end = np.array([])
            loess_rr_end = np.array([])
        tab_final = tab_codes
    else:
        loess_rpeaks_end, loess_seg_end, loess_rr_end = np.array([]), np.array([]), np.array([])
        loess_data, loess_fitted = np.array([]), np.array([])
        paso1_beats = np.array([])
        tab_final = np.array([[1234, 1, 11]])
        
    return {
        "loessRPeaksEnd": loess_rpeaks_end, # picchi R (che hanno superato lo scarto minimo)
        "loessSegEnd": loess_seg_end,       # dimensioni di segmentazione
        "loessRREnd": loess_rr_end,         # distanze RR   
        "tabFinal": tab_final,              # tab_codes
        "loessData": loess_data,            # ECG dopo rimozione baseline wander
        "loessFitted": loess_fitted,        # baseline wander
        "paso1": paso1_beats                # picchi dopo drop_anno1_app
    }

def lead_pre_multi_app(obj_pre_pantom, annos_ref, seg_ref, freq):
    # R equivalent: leadPreMulti_app
    # Evaluates artifacts against global median annotations
    loess_rpeaks = annos_ref
    loess_seg = seg_ref
    loess_rr = np.diff(loess_rpeaks) if len(loess_rpeaks) > 1 else []
    
    if len(annos_ref) >= 3 and np.sum(obj_pre_pantom["loessData"] != 0) > 0.75 * len(obj_pre_pantom["loessData"]):
        loess_data = obj_pre_pantom["loessData"]
        loess_fitted = obj_pre_pantom["loessFitted"]
        
        tab_codes = np.zeros((len(loess_rpeaks), 3), dtype=int)
        tab_codes[:, 0] = 1234
        tab_codes[:, 1] = np.arange(1, len(loess_rpeaks) + 1)
        tab_codes[:, 2] = 0
        
        # TODO: da implementare? Al momento sono stub
        paso2_beats, paso2_idx = drop_anno2_app(loess_data, loess_rpeaks, loess_rr, loess_seg, loess_rpeaks, freq)
        paso3_beats, paso3_idx = drop_anno3_app(loess_rpeaks, loess_rr, loess_seg, loess_rpeaks)
        paso5_beats, paso5_idx = drop_anno5_app(loess_data, loess_rpeaks, loess_rr, loess_seg, loess_rpeaks)
        
        union = np.intersect1d(paso2_beats, np.intersect1d(paso3_beats, paso5_beats)) if len(paso5_beats) > 0 else np.intersect1d(paso2_beats, paso3_beats)
        
        if len(union) > 0:
            loess_rpeaks_end = np.array(union)
            indices = [list(loess_rpeaks).index(u) for u in union if u in loess_rpeaks]
            loess_seg_end = np.array([loess_seg[i] for i in indices]) if len(loess_seg) > 0 else []
            # Simplified RR end
            loess_rr_end = loess_rpeaks_end
    else:
        loess_rpeaks_end, loess_seg_end, loess_rr_end = np.array([]), np.array([]), np.array([])
        tab_codes = np.array([[1234, 1, 10]])
        
    return {
        # loessRPeaksEnd: picchi R (che hanno superato lo scarto minimo)
        # loessSegEnd: segmenti ECG ([inizio, fine])
        # loessRREnd: distanze RR
        # tabFinal: tab_codes
        "loessRPeaksEnd": loess_rpeaks_end,
        "loessSegEnd": loess_seg_end,
        "loessRREnd": loess_rr_end,
        "tabFinal": tab_codes
    }

# =======================================================================
# TOP-LEVEL ORCHESTRATOR
# =======================================================================

def give_preprocessing_git(data_in, freq_hz):
    """
    R equivalent: givePreprocessing_git
    data_in: np.ndarray shape (n_leads, n_samples)
    """    

    n_leads = data_in.shape[0]
    pre_results, annos_list, leads_list = [], [], []
    
    # 1. Itera sulle derivazioni e applica LOESS + Pantompkins (individuazione picchi R sulle singole derivazioni)
    for i in range(n_leads):
        res = lead_pre_pan_tom_app(data_in[i], freq_hz)
        pre_results.append(res)
        
        peaks = res["loessRPeaksEnd"]
        if len(peaks) > 0:
            annos_list.extend(peaks)
            leads_list.extend([i] * len(peaks))
            
    # annos_list: lista delle annotazioni (picchi R)
    # leads_list: lista delle derivazioni associate ad ogni picco R
    # TODO: refactoring: usare una lista di lista, dove l'indice esterno fa riferimento alla derivazione    

    # 2. Median Alignment e merge (per derivazione) - crea unico file annotazioni per tutto l'ecg
    anno_leads = np.array([annos_list, leads_list])
    if anno_leads.shape[1] > 0:
        sort_idx = np.argsort(anno_leads[0])
        # [[  151   348  1173 ... 36696 37542 38374][    0     0     0 ...    11    11    11]]
        anno_leads = anno_leads[:, sort_idx]
        
        # dopo argsort, ordina le annotazioni e le derivazioni in base alle annotazioni:
        # # [[  151   151  151 ... 36696 37542 38374][    0     4     9 ...    1    3    7]]
        # dunque anno_leads[0] è un unico grande array con le annotazioni dei picchi associate a TUTTE le 
        # derivazioni in maniera crescente

        # annos_pac -> array di indici dei picchi R (globali)
        # leads_pac -> array di indici delle derivazioni associati ad ogni picco R globale
        # seg_pac -> array di indici dei segmenti ECG ([inizio, fine])
        annos_pac, leads_pac, seg_pac = give_anno_med_app(anno_leads, data_in.shape[1], freq_hz)

        # sanity check
        if len(annos_pac) == len(leads_pac) == len(seg_pac): print(f"identificati {len(annos_pac)} complessi QRS")
        else: raise ValueError("Errore nel preprocessing: numero di annotazioni non corrispondente")
            
        # 3. Pulizia delle singole derivazioni sulla base degli indici globali (annos_pac)
        pos_results = []
        for i in range(n_leads):
            pos_res = lead_pre_multi_app(pre_results[i], annos_pac, seg_pac, freq_hz)
            pos_results.append(pos_res)
        pos_results = np.array(pos_results)

        # segnale detrendato (senza componente a bassa frequenza)   
        m_detrend = np.vstack([r["loessData"] if len(r["loessData"])>0 else np.full(data_in.shape[1], np.nan) for r in pre_results])
        error_preprocessing = False
    else:
        pos_results = []
        m_detrend = np.full((n_leads, data_in.shape[1]), np.nan)
        error_preprocessing = True
        
    # pos_results: lista di dizionari, uno per ogni derivazione, ciascuno contenente:
    #   loessRPeaksEnd: picchi R (che hanno superato lo scarto minimo)
    #   loessSegEnd: segmenti ECG ([inizio, fine])
    #   loessRREnd: distanze RR **(UGUALE A loessRPeaksEnd???)**
    #   tabFinal: tab_codes
    # m_detrend: matrice numpy contenente i segnali ECG detrendati (PRIMA della pulizia )
    # error_preprocessing: booleano che indica se c'è stato un errore nel preprocessing
    return pos_results, m_detrend, error_preprocessing
