import numpy as np
from preprocessing.utils import compute_global_median_peaks
from preprocessing.lead_processor import LeadProcessor


class ECGPreprocessor:
    """
    Classe per la gestione della pipeline di preprocessing del segnale ECG.
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
        Esegue la pipeline completa di preprocessing.
        Args:
            data_in: Segnale ECG con shape (n_leads, n_samples).
            
        Returns:
            A tuple containing:
                - pos_results: Dizionario con i risultati del preprocessing per ogni lead.
                - m_detrend: Segnale ECG detrendato con shape (n_leads, n_samples).
                - error_preprocessing: Booleano che indica un errore nel preprocessing.
        """
        n_leads, n_samples = data_in.shape[0], data_in.shape[1]
        
        # 1) Elaborazione individuale di ogni derivazione
        self._initial_lead_processing(data_in)
        
        # 2) Allineamento globale con picchi di R medi
        self._global_alignment(n_samples)
        
        # 3) Filtraggio artefatti
        if not self.error_preprocessing:
            self._artifact_filtering(n_samples)
        else:
            self.pos_results = []
            self.m_detrend = np.full((n_leads, n_samples), np.nan)
            
        return self.pos_results, self.m_detrend, self.error_preprocessing

    def _initial_lead_processing(self, data_in):
        """
        Rimuove la baseline (componenti a bassa frequenza) e rileva i picchi QRS iniziali per ogni derivazione.
        """
        self.lead_processors = []
        for i in range(data_in.shape[0]):
            lp = LeadProcessor(data_in[i], i, self.freq)
            lp.preprocess_single_lead()
            self.lead_processors.append(lp)

    def _global_alignment(self, n_samples):
        """
        Allineamento globale dei picchi di R.
        """
        all_peaks_data = []
        for lp in self.lead_processors:
            peaks = lp.filter1_beats if len(lp.filter1_beats) > 0 else lp.local_rpeaks
            for p in peaks: all_peaks_data.append((p, lp.lead_idx))

        # all_peaks_data è una lista di tuple: (rpeak, lead_idx) ordinata per rpeak        
        if all_peaks_data:
            all_peaks_data.sort(key=lambda x: x[0])        
            self.global_rpeaks, _, self.global_segments = compute_global_median_peaks(all_peaks_data, n_samples, self.freq)
            self.error_preprocessing = False
        else:
            self.error_preprocessing = True


    def _artifact_filtering(self, n_samples):
        """
        Affina le annotazioni di ogni derivazione basandosi sul set mediano globale.
        """
        self.pos_results = []
        for lp in self.lead_processors:
            lp.filter_artifacts(self.global_rpeaks, self.global_segments)
            self.pos_results.append(lp.to_dict())
            
        self.pos_results = np.array(self.pos_results)
        
        # Assembla la matrice di segnale detrended
        self.m_detrend = np.vstack([
            lp.loess_data if len(lp.loess_data) > 0 else np.full(n_samples, np.nan) 
            for lp in self.lead_processors
        ])