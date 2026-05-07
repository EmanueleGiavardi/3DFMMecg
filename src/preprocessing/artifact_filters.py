import numpy as np

class ArtifactFilters:
    @staticmethod
    def reject_rr_amplitude(data, anno, rr, seg, freq):
        """
        Step 1: Reject beats based on RR interval consistency and local amplitude.
        (Original name: drop_anno1_app)
        """
        if len(anno) <= 1:
            return anno.tolist() if isinstance(anno, np.ndarray) else list(anno), []
            
        anno_new = np.array(anno)
        rr_new = np.array(rr)
        seg_new = np.array(seg)
        anno_orig = np.array(anno)
        
        rr_threshold = (2/3) * np.median(rr_new)
        
        def get_peak_distances(a):
            return np.diff(a) if len(a) > 1 else np.array([])
            
        peak_diffs = get_peak_distances(anno_new)
        
        def check_should_continue(rr_arr, seg_arr, diff_arr):
            if len(rr_arr) == 0: return False
            cond1 = np.sum(rr_arr < rr_threshold) > 0
            cond2 = np.sum(diff_arr > (4/3)*np.median(rr_arr)) >= 1
            cond3_a = len(seg_arr) > (4/3)*(len(data)/freq)
            if len(rr_arr) > 1 and np.mean(rr_arr) != 0:
                std_rr = np.std(rr_arr, ddof=1)
                cond3_b = (std_rr / np.mean(rr_arr)) > 0.2
            else:
                cond3_b = False
            return cond1 or cond2 or (cond3_a and cond3_b)

        should_continue = check_should_continue(rr_new, seg_new, peak_diffs)
        iteration_stage = 1
        
        while should_continue:
            reject_mask = rr_new < rr_threshold
            
            has_large_diffs = np.sum(peak_diffs > (4/3)*np.median(rr_new)) >= 1
            has_no_short_rr = np.sum(rr_new < rr_threshold) == 0
            
            if has_large_diffs and has_no_short_rr and iteration_stage == 1:
                reject_mask = peak_diffs > (4/3)*np.median(rr_new)
                iteration_stage = 2
            else:
                is_long_recording = len(seg_new) > (4/3)*(len(data)/freq)
                if len(rr_new) > 1 and np.mean(rr_new) != 0:
                    high_variability = (np.std(rr_new, ddof=1) / np.mean(rr_new)) > 0.2
                else:
                    high_variability = False
                if is_long_recording and high_variability:
                    reject_mask = np.array([True]*(len(seg_new)-2) + [False])
                    
            target_indices = np.where(reject_mask)[0]
            if len(target_indices) == 0: break
                
            # Range analysis around the peaks
            window = int(np.ceil(0.02*freq))
            range_current_start = anno_new[target_indices] - window
            range_current_end = anno_new[target_indices] + window
            range_next_start = anno_new[target_indices+1] - window
            range_next_end = anno_new[target_indices+1] + window
            
            def get_range_amp(d, start, end):
                s = max(0, int(start))
                e = min(len(d), int(end)+1)
                if s >= e: return 0
                return np.max(d[s:e]) - np.min(d[s:e])

            amp_current = np.array([get_range_amp(data, range_current_start[i], range_current_end[i]) for i in range(len(target_indices))])
            amp_next = np.array([get_range_amp(data, range_next_start[i], range_next_end[i]) for i in range(len(target_indices))])
            
            to_merge = []
            for i, idx in enumerate(target_indices):
                s = max(0, int(seg_new[idx, 0]))
                e = min(len(data), int(seg_new[idx, 1])+1)
                segment_data = data[s:e] if s < e else np.array([0])
                segment_amp = np.max(segment_data) - np.min(segment_data)
                
                c1 = abs(amp_current[i] - amp_next[i]) > 0.25 * segment_amp
                c2 = segment_amp > 2 * abs(amp_current[i]) and segment_amp > 2 * abs(amp_next[i])
                c3 = abs(amp_current[i] - amp_next[i]) < 0.25 * segment_amp
                
                if (c1 or (c2 and c3)) and (seg_new[idx+1, 0] - seg_new[idx, 1] < (1/3)*np.median(rr_new)):
                    if amp_current[i] > amp_next[i]:
                        to_merge.append(idx + 1)
                    else:
                        to_merge.append(idx)
                        
            to_merge = np.unique(to_merge)
            if len(to_merge) > 0:
                anno_new = np.delete(anno_new, to_merge)
                seg_new = np.delete(seg_new, to_merge, axis=0)
                
            rr_new = get_peak_distances(anno_new)
            
            to_delete = []
            if len(anno_new) > 2 and iteration_stage == 1:
                for j in range(1, len(anno_new)-1):
                    win_curr_s, win_curr_e = anno_new[j] - window, anno_new[j] + window
                    win_next_s, win_next_e = anno_new[j+1] - window, anno_new[j+1] + window
                    win_prev_s, win_prev_e = anno_new[j-1] - window, anno_new[j-1] + window
                    
                    def get_max_in_range(d, start, end):
                        s, e = max(0, int(start)), min(len(d), int(end)+1)
                        return np.max(d[s:e]) if s < e else -np.inf
                        
                    val_curr = get_max_in_range(data, win_curr_s, win_curr_e)
                    val_prev = get_max_in_range(data, win_prev_s, win_prev_e)
                    val_next = get_max_in_range(data, win_next_s, win_next_e)
                    
                    is_much_higher_than_prev = (2 * val_prev < val_curr)
                    is_much_higher_than_next = (2 * val_next < val_curr)
                    
                    seg_len = min(len(data), int(seg_new[j, 1])) - max(0, int(seg_new[j, 0]))
                    std_rr = np.std(rr_new, ddof=1) if len(rr_new) > 1 else 0
                    mean_rr = np.mean(rr_new) if len(rr_new) > 0 else 1
                    
                    if (is_much_higher_than_prev or is_much_higher_than_next) and seg_len < (2/3)*freq and (std_rr / mean_rr) < 0.2:
                        if is_much_higher_than_prev: to_delete.append(j-1)
                        else: to_delete.append(j+1)
                            
                if len(to_delete) > 0:
                    to_delete = np.unique(to_delete)
                    anno_new = np.delete(anno_new, to_delete)
                    seg_new = np.delete(seg_new, to_delete, axis=0)
                    
            iteration_stage = 2
            rr_new = get_peak_distances(anno_new)
            if len(rr_new) > 0:
                rr_threshold = (2/3)*np.median(rr_new)
            
            if len(anno_new) <= 1:
                should_continue = False
            else:
                peak_diffs = get_peak_distances(anno_new)
                should_continue = np.sum(rr_new < rr_threshold) > 0
                if len(to_merge) == 0 and len(to_delete) == 0:
                    should_continue = False
                    
        rejected_indices = [i for i, a in enumerate(anno_orig) if a not in anno_new]
        return anno_new.tolist(), rejected_indices

    @staticmethod
    def reject_excursion(data, anno, rr, seg, anno_orig, freq):
        """
        Step 2: Reject beats based on signal excursion (identifying flat lines or noise).
        (Original name: drop_anno2_app)
        """
        anno_new = np.array(anno)
        seg_new = np.array(seg)
        anno_orig = np.array(anno_orig)
        
        if len(anno_new) <= 1:
            return anno_new.tolist(), []
            
        condition_1 = False
        condition_2 = False
        
        def get_boundary_samples(seg_n, d):
            if len(seg_n) == 0: return np.array([]), np.array([])
            # Start of first segment
            s1, e1 = max(0, int(seg_n[0, 0])), min(len(d), int(seg_n[0, 1]))
            end_a = max(s1 + int(np.ceil(0.02*(e1 - s1))), 1) if e1 > s1 else s1
            samples_a = d[s1:end_a]
            
            # End of last segment
            s_end, e_end = max(0, int(seg_n[-1, 0])), min(len(d), int(seg_n[-1, 1]))
            window = int(np.ceil(0.02*(e_end - s_end))) if e_end > s_end else 0
            start_b = max(0, e_end - window)
            samples_b = d[start_b:e_end]
            return samples_a, samples_b

        samples_a, samples_b = get_boundary_samples(seg_new, data)
        
        # Initial segment range analysis
        start_idx, end_idx = max(0, int(seg_new[0, 0])), min(len(data), int(seg_new[0, 1]))
        total_len = end_idx - start_idx
        range_start = start_idx - 1 + int(np.ceil(0.15 * total_len))
        range_end = end_idx + 1 - int(np.ceil(0.15 * total_len))
        
        data_norm = data[start_idx:end_idx]
        if len(data_norm) > 0:
            val_min, val_max = np.min(data_norm), np.max(data_norm)
            idx_min, idx_max = np.argmin(data_norm), np.argmax(data_norm)
            excursion_1 = abs(val_min - val_max)
            if idx_min <= (range_start - start_idx) or idx_max <= (range_start - start_idx):
                condition_1 = True
        else:
            excursion_1 = 0

        excursions = []
        for j in range(1, len(seg_new)):
            s, e = max(0, int(seg_new[j, 0])), min(len(data), int(seg_new[j, 1]))
            dn = data[s:e]
            excursions.append(abs(np.min(dn) - np.max(dn)) if len(dn) > 0 else 0)
                
        if len(excursions) > 0:
            median_excursion = np.median(excursions)
            if excursion_1 > (2/3)*median_excursion or excursion_1 < (2/3)*median_excursion:
                if condition_1: condition_2 = True
                
        def get_sd_mean(arr):
            if len(arr) < 2: return 0, 1
            return np.std(arr, ddof=1), np.mean(arr)
            
        def check_should_continue(sn, an, samples_a, samples_b, cond2):
            if len(sn) <= 1: return False
            med_seg_len = np.median(sn[:, 1] - sn[:, 0])
            c1 = (sn[0, 1] - sn[0, 0]) < (0.9 * med_seg_len) and len(an) > 1
            sd_a, mean_a = get_sd_mean(samples_a)
            c2 = (sd_a / abs(mean_a)) < 0.01 if mean_a != 0 else False
            c3 = (sd_a / abs(mean_a)) < 0.02 and len(an) >= 0.04*freq if mean_a != 0 else False
            return c1 or c2 or c3 or cond2
            
        should_continue = check_should_continue(seg_new, anno_new, samples_a, samples_b, condition_2)
        
        while should_continue:
            anno_new = anno_new[1:]
            seg_new = seg_new[1:]
            if len(anno_new) <= 1:
                should_continue = False
            else:
                med_seg_len = np.median(seg_new[:, 1] - seg_new[:, 0])
                should_continue = (seg_new[0, 1] - seg_new[0, 0]) < (0.85 * med_seg_len) and len(anno_new) > 1

        if len(anno_new) > 1:
            med_seg_len = np.median(seg_new[:, 1] - seg_new[:, 0])
            sd_b, mean_b = get_sd_mean(samples_b)
            c1 = (seg_new[-1, 1] - seg_new[-1, 0]) < (0.9 * med_seg_len)
            c2 = (sd_b / abs(mean_b)) < 0.01 if mean_b != 0 else False
            c3 = (sd_b / abs(mean_b)) < 0.02 and len(anno_new) >= 0.04*freq if mean_b != 0 else False
            should_continue = c1 or c2 or c3
            
        while should_continue:
            anno_new = anno_new[:-1]
            seg_new = seg_new[:-1]
            if len(anno_new) <= 1:
                should_continue = False
            else:
                med_seg_len = np.median(seg_new[:, 1] - seg_new[:, 0])
                should_continue = (seg_new[-1, 1] - seg_new[-1, 0]) < (0.85 * med_seg_len) and len(anno_new) > 1
                
        rejected_indices = [i for i, a in enumerate(anno_orig) if a not in anno_new]
        return anno_new.tolist(), rejected_indices

    @staticmethod
    def reject_peak_center(anno, rr, seg, anno_orig):
        """
        Step 3: Reject beats where the R-peak is not correctly centered within its segment.
        (Original name: drop_anno3_app)
        """
        anno_new = np.array(anno)
        seg_new = np.array(seg)
        anno_orig = np.array(anno_orig)
        
        if len(anno_new) > 1 and len(seg_new) > 0:
            seg_len = seg_new[:, 1] - seg_new[:, 0]
            relative_pos = anno_new - seg_new[:, 0] + 1
            is_too_early = (seg_len * 0.35) > relative_pos
            is_too_late = relative_pos > (0.45 * seg_len)
            to_reject = np.where(is_too_early | is_too_late)[0]
            if len(to_reject) > 0:
                anno_new = np.delete(anno_new, to_reject)
                seg_new = np.delete(seg_new, to_reject, axis=0)
                
        rejected_indices = [i for i, a in enumerate(anno_orig) if a not in anno_new]
        return anno_new.tolist(), rejected_indices

    @staticmethod
    def reject_baseline_trend(data, anno, rr, seg, anno_orig):
        """
        Step 5: Reject beats based on significant baseline wander trends between start and end.
        (Original name: drop_anno5_app)
        """
        anno_new = np.array(anno)
        seg_new = np.array(seg)
        anno_orig = np.array(anno_orig)
        
        if len(anno_new) > 1:
            diff_start_end, amplitude = np.zeros(len(anno_new)), np.zeros(len(anno_new))
            is_bad_trend = np.zeros(len(anno_new), dtype=bool)
            to_review = []
            
            for j in range(len(anno_new)):
                start, end = max(0, int(seg_new[j, 0])), min(len(data), int(seg_new[j, 1]))
                beat_len = end - start
                
                # Check consistency of median values at the beginning and end of the segment
                data_segment = data[start:end]
                if len(data_segment) > 0:
                    edge_size = int(np.ceil(0.05 * beat_len))
                    val_start = np.median(data_segment[:edge_size]) if edge_size > 0 else 0
                    val_end = np.median(data_segment[-edge_size:]) if edge_size > 0 else 0
                    
                    diff_start_end[j] = abs(val_end - val_start)
                    amplitude[j] = abs(np.max(data_segment) - np.min(data_segment))
                    
                    is_bad_trend[j] = diff_start_end[j] > 0.15 * amplitude[j]
                    if is_bad_trend[j]: to_review.append(j)
                
            # If too many beats are bad, reject the whole lead
            if np.sum(is_bad_trend) >= 4:
                to_review = list(range(len(anno_new)))
                
            if len(to_review) > 0:
                to_reject = np.unique(np.where(diff_start_end > amplitude)[0].tolist() + to_review)
                anno_new = np.delete(anno_new, to_reject)
                
        rejected_indices = [i for i, a in enumerate(anno_orig) if a not in anno_new]
        return anno_new.tolist(), rejected_indices
