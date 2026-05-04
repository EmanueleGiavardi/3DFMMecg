import numpy as np

class ArtifactFilters:
    @staticmethod
    def reject_rr_amplitude(data, anno, rr, seg, freq):
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

    @staticmethod
    def reject_excursion(data, anno, rr, seg, anno_orig, freq):
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

    @staticmethod
    def reject_peak_center(anno, rr, seg, anno_orig):
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

    @staticmethod
    def reject_baseline_trend(data, anno, rr, seg, anno_orig):
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

