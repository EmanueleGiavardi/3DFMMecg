import numpy as np
from skmisc.loess import loess
from preprocessing.utils import detect_rpeaks_pantom, compute_segment_ecg_beats
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
        
        # State
        self.loess_data = np.array([])
        self.loess_fitted = np.array([])
        self.local_rpeaks = np.array([])
        self.local_seg = np.array([])
        self.local_rr = np.array([])
        
        self.final_rpeaks = np.array([])
        self.final_seg = np.array([])
        self.final_rr = np.array([])
        self.tab_codes = np.array([[1234, 1, 11]])
        self.paso1_beats = np.array([])
        self.is_valid = False

    def preprocess_single_lead(self):
        if np.sum(self.raw_data != 0) <= 0.75 * self.n_samples:
            return self

        self.is_valid = True
        dd = self.raw_data.copy()
        
        l = loess(np.arange(self.n_samples), dd, span=0.2, degree=2)
        l.fit()
        self.loess_fitted = l.outputs.fitted_values
        self.loess_data = dd - self.loess_fitted
        
        try:
            peaks_sec = detect_rpeaks_pantom(self.loess_data, self.freq)
            self.local_rpeaks = (peaks_sec * self.freq) - 1
        except:
            self.local_rpeaks = np.array([])

        while len(self.local_rpeaks) <= 2:
            q1 = np.quantile(dd, 0.9999)
            if np.max(dd) >= q1: dd[dd >= q1] = np.median(dd)
            q2 = np.quantile(dd, 0.0001)
            if np.min(dd) <= q2: dd[dd <= q2] = np.median(dd)
                
            l = loess(np.arange(self.n_samples), dd, span=0.2, degree=2)
            l.fit()
            self.loess_fitted = l.outputs.fitted_values
            self.loess_data = dd - self.loess_fitted
            try:
                peaks_sec = detect_rpeaks_pantom(self.loess_data, self.freq)
                self.local_rpeaks = (peaks_sec * self.freq) - 1
            except:
                break
                
        self.local_seg = compute_segment_ecg_beats(self.loess_data, self.local_rpeaks, self.freq)
        self.local_rr = np.diff(self.local_rpeaks) if len(self.local_rpeaks) > 1 else np.array([])
        
        self.tab_codes = np.zeros((len(self.local_rpeaks), 3), dtype=int)
        self.tab_codes[:, 0] = 1234
        self.tab_codes[:, 1] = np.arange(1, len(self.local_rpeaks) + 1)
        self.tab_codes[:, 2] = 0
        
        paso1_beats, _ = ArtifactFilters.reject_rr_amplitude(self.loess_data, self.local_rpeaks, self.local_rr, self.local_seg, self.freq)
        self.paso1_beats = np.array(paso1_beats)
        
        return self

    def filter_artifacts(self, global_peaks, global_segments):
        if not self.is_valid or len(global_peaks) < 3:
            self.tab_codes = np.array([[1234, 1, 10]])
            return self

        global_rr = np.diff(global_peaks) if len(global_peaks) > 1 else np.array([])
        
        self.tab_codes = np.zeros((len(global_peaks), 3), dtype=int)
        self.tab_codes[:, 0] = 1234
        self.tab_codes[:, 1] = np.arange(1, len(global_peaks) + 1)
        self.tab_codes[:, 2] = 0
        
        paso2_beats, _ = ArtifactFilters.reject_excursion(self.loess_data, global_peaks, global_rr, global_segments, global_peaks, self.freq)
        paso3_beats, _ = ArtifactFilters.reject_peak_center(global_peaks, global_rr, global_segments, global_peaks)
        paso5_beats, _ = ArtifactFilters.reject_baseline_trend(self.loess_data, global_peaks, global_rr, global_segments, global_peaks)
        
        if len(paso5_beats) > 0:
            union = np.intersect1d(paso2_beats, np.intersect1d(paso3_beats, paso5_beats))
        else:
            union = np.intersect1d(paso2_beats, paso3_beats)
            
        if len(union) > 0:
            self.final_rpeaks = np.array(union)
            indices = [list(global_peaks).index(u) for u in union if u in global_peaks]
            self.final_seg = np.array([global_segments[i] for i in indices]) if len(global_segments) > 0 else np.array([])
            self.final_rr = self.final_rpeaks
            
        return self
        
    def to_dict(self):
        return {
            "loessRPeaksEnd": self.final_rpeaks,
            "loessSegEnd": self.final_seg,
            "loessRREnd": self.final_rr,
            "tabFinal": self.tab_codes,
            "loessData": self.loess_data,
            "loessFitted": self.loess_fitted,
            "paso1": self.paso1_beats
        }
