import numpy as np
from skmisc.loess import loess
from preprocessing.utils import detect_rpeaks, compute_segment_ecg_beats
from preprocessing.artifact_filters import ArtifactFilters


class LeadProcessor:
    """
    Class to preprocess a single lead of ECG.
    """
    def __init__(self, lead_data: np.ndarray, lead_idx: int, freq_hz: int):
        self.raw_data = np.nan_to_num(lead_data, copy=True)
        self.lead_idx = lead_idx
        self.freq = freq_hz
        self.n_samples = len(self.raw_data)
        self.patient_code = 1234
        
        self.loess_data = np.array([])
        self.loess_fitted = np.array([])

        self.loess_rpeaks_end = np.array([])
        self.loess_seg_end = np.array([])
        self.loess_rr_end = np.array([])
        
        self.tab_codes = np.array([[self.patient_code, 1, 11]])
        self.filter1_beats = np.array([])
        
        self.is_valid = False

    def preprocess_single_lead(self):
        """
        Equivalent to preprocess_single_lead function in R.
        """
        if np.sum(self.raw_data != 0) <= 0.75 * self.n_samples:
            return self

        self.is_valid = True
        dd = self.raw_data.copy()
        
        # Baseline removal
        l = loess(np.arange(self.n_samples), dd, span=0.2, degree=2)
        l.fit()
        self.loess_fitted = l.outputs.fitted_values
        self.loess_data = dd - self.loess_fitted
        
        # Initial R-peak detection
        try:
            peaks_sec = detect_rpeaks(self.loess_data, self.freq)
            # Match R logic: (sec * freq) - 1
            current_rpeaks = (peaks_sec * self.freq) - 1
        except:
            current_rpeaks = np.array([])

        # Handle anomalous signals by median-clipping and re-fitting LOESS
        while len(current_rpeaks) <= 2:
            q1 = np.quantile(dd, 0.9999)
            if np.max(dd) >= q1: dd[dd >= q1] = np.median(dd)
            q2 = np.quantile(dd, 0.0001)
            if np.min(dd) <= q2: dd[dd <= q2] = np.median(dd)
                
            l = loess(np.arange(self.n_samples), dd, span=0.2, degree=2)
            l.fit()
            self.loess_fitted = l.outputs.fitted_values
            self.loess_data = dd - self.loess_fitted
            try:
                peaks_sec = detect_rpeaks(self.loess_data, self.freq)
                current_rpeaks = (peaks_sec * self.freq) - 1
            except:
                break
        
        # Initial segmentation
        current_seg = compute_segment_ecg_beats(self.loess_data, current_rpeaks, self.freq)
        current_rr = np.diff(current_rpeaks) if len(current_rpeaks) > 1 else np.array([])
        
        # Initial tab codes
        self.tab_codes = np.zeros((len(current_rpeaks), 3), dtype=int)
        self.tab_codes[:, 0] = self.patient_code
        self.tab_codes[:, 1] = np.arange(1, len(current_rpeaks) + 1)
        self.tab_codes[:, 2] = 0

        # Paso 1: local RR and amplitude filtering
        paso1_beats, _ = ArtifactFilters.reject_rr_amplitude(self.loess_data, current_rpeaks, current_rr, current_seg, self.freq)
        self.filter1_beats = np.array(paso1_beats)
        
        if len(self.filter1_beats) > 0:
            self.loess_rpeaks_end = self.filter1_beats
            self.loess_seg_end = compute_segment_ecg_beats(self.loess_data, self.filter1_beats, self.freq)
            # In Phase 1 loess_rr_end contains RR intervals
            self.loess_rr_end = np.diff(self.filter1_beats) if len(self.filter1_beats) > 1 else np.array([])
        else:
            self.loess_rpeaks_end = np.array([])
            self.loess_seg_end = np.array([])
            self.loess_rr_end = np.array([])
            
        return self

    def filter_artifacts(self, global_peaks, global_segments):
        """
        Equivalent to filter_lead_artifacts function in R.
        Refines annotations based on global consensus set.
        """
        if not self.is_valid or len(global_peaks) < 3:
            self.loess_rpeaks_end = np.array([])
            self.loess_seg_end = np.array([])
            self.loess_rr_end = np.array([])
            self.tab_codes = np.array([[self.patient_code, 1, 10]])
            return self

        current_rr = np.diff(global_peaks) if len(global_peaks) > 1 else np.array([])
        
        # Update tab codes for global set
        self.tab_codes = np.zeros((len(global_peaks), 3), dtype=int)
        self.tab_codes[:, 0] = self.patient_code
        self.tab_codes[:, 1] = np.arange(1, len(global_peaks) + 1)
        self.tab_codes[:, 2] = 0
        
        # Apply artifact filters (Step 2, 3, 5)
        f2_beats, _ = ArtifactFilters.reject_excursion(self.loess_data, global_peaks, current_rr, global_segments, global_peaks, self.freq)
        f3_beats, _ = ArtifactFilters.reject_peak_center(global_peaks, current_rr, global_segments, global_peaks)
        f5_beats, _ = ArtifactFilters.reject_baseline_trend(self.loess_data, global_peaks, current_rr, global_segments, global_peaks)
        
        if len(f5_beats) > 0:
            union = np.intersect1d(f2_beats, np.intersect1d(f3_beats, f5_beats))
        else:
            union = np.intersect1d(f2_beats, f3_beats)
            
        if len(union) > 0:
            self.loess_rpeaks_end = np.array(union)
            indices = [list(global_peaks).index(u) for u in union if u in global_peaks]
            self.loess_seg_end = np.array([global_segments[i] for i in indices]) if len(global_segments) > 0 else np.array([])
            # In Phase 2 loess_rr_end contains the peaks (match R inconsistency)
            self.loess_rr_end = self.loess_rpeaks_end
        else:
            self.loess_rpeaks_end = np.array([])
            self.loess_seg_end = np.array([])
            self.loess_rr_end = np.array([])
            
        return self
        
    def to_dict(self):
        return {
            "loessRPeaksEnd": self.loess_rpeaks_end,
            "loessSegEnd": self.loess_seg_end,
            "loessRREnd": self.loess_rr_end,
            "tabFinal": self.tab_codes,
            "loessData": self.loess_data,
            "loessFitted": self.loess_fitted,
            "filter1_beats": self.filter1_beats
        }
