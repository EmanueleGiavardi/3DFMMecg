import numpy as np
from preprocessing.utils import compute_global_median_peaks
from preprocessing.lead_processor import LeadProcessor


class ECGPreprocessor:
    """
    Top-level class to orchestrate the ECG preprocessing pipeline.
    """
    def __init__(self, freq_hz: int = 500):
        self.freq = freq_hz
        self.lead_processors = []
        self.global_rpeaks = None
        self.global_segments = None
        self.m_detrend = None
        self.pos_results = []
        self.error_preprocessing = False
        
    def run(self, data_in: np.ndarray):
        """
        Executes the full preprocessing pipeline.
        Args:
            data_in: Input ECG data with shape (n_leads, n_samples).
            
        Returns:
            A tuple containing:
                - pos_results: List of dictionaries with preprocessing results for each lead.
                - m_detrend: Detrended ECG data with shape (n_leads, n_samples).
                - error_preprocessing: Boolean indicating whether preprocessing was successful.
        """
        n_leads, n_samples = data_in.shape[0], data_in.shape[1]
        
        # 1: Individual Lead Processing
        self._initial_lead_processing(data_in)
        
        # 2: Global Median Alignment
        self._global_alignment(n_samples)
        
        # 3: Artifact Filtering
        if not self.error_preprocessing:
            self._artifact_filtering(n_samples)
        else:
            self.pos_results = []
            self.m_detrend = np.full((n_leads, n_samples), np.nan)
            
        return self.pos_results, self.m_detrend, self.error_preprocessing

    def _initial_lead_processing(self, data_in):
        """
        Removes baseline (low frequency components) and detects initial QRS for each lead.
        """
        self.lead_processors = []
        for i in range(data_in.shape[0]):
            lp = LeadProcessor(data_in[i], i, self.freq)
            lp.preprocess_single_lead()
            self.lead_processors.append(lp)

    def _global_alignment(self, n_samples):
        """
        Consolidates R-peaks from all leads into a median global set.
        """
        
        all_peaks_data = []
        for lp in self.lead_processors:
            peaks = lp.paso1_beats if len(lp.paso1_beats) > 0 else lp.local_rpeaks
            for p in peaks:
                all_peaks_data.append((p, lp.lead_idx))

        # all_peaks_data is list of tuple: (rpeak, lead_idx) sorted by rpeak        

        if all_peaks_data:
            all_peaks_data.sort(key=lambda x: x[0])
            anno_leads = np.array(all_peaks_data).T
        
            self.global_rpeaks, _, self.global_segments = compute_global_median_peaks(anno_leads, n_samples, self.freq)
            self.error_preprocessing = False
        else:
            self.error_preprocessing = True


    def _artifact_filtering(self, n_samples):
        """
        Refines each lead's annotations based on the global median set.
        """
        self.pos_results = []
        for lp in self.lead_processors:
            lp.filter_artifacts(self.global_rpeaks, self.global_segments)
            self.pos_results.append(lp.to_dict())
            
        self.pos_results = np.array(self.pos_results)
        
        # Assemble the detrended signal matrix
        self.m_detrend = np.vstack([
            lp.loess_data if len(lp.loess_data) > 0 else np.full(n_samples, np.nan) 
            for lp in self.lead_processors
        ])