import numpy as np
import pandas as pd

class VCGComparator:
    """Classe dedicata al calcolo delle metriche di errore morfologico e spaziale."""
    
    def __init__(self, vcg_ref: dict, vcg_test: dict):
        self.vcg_ref = vcg_ref
        self.vcg_test = vcg_test
        self.ref_array = np.column_stack((vcg_ref['X'], vcg_ref['Y'], vcg_ref['Z']))
        self.test_array = np.column_stack((vcg_test['X'], vcg_test['Y'], vcg_test['Z']))

    def _get_peak_vector(self, trajectory: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(trajectory, axis=1)
        peak_idx = np.argmax(norms)
        return trajectory[peak_idx]

    @staticmethod
    def _compute_qrs_t_angle_for_array(arr: np.ndarray) -> float:
        """
        Trova l'angolo QRS-T per un singolo array VCG.
        """
        norms = np.linalg.norm(arr, axis=1)
        n_samples = len(norms)
        
        # Trova il picco QRS
        idx_qrs = np.argmax(norms)
        v_qrs = arr[idx_qrs]
        
        # Trova il picco T (nell'ultima parte del battito)
        start_t_search = int(n_samples * 0.6)
        if start_t_search <= idx_qrs: start_t_search = idx_qrs + int(n_samples * 0.1)
            
        idx_t_relative = np.argmax(norms[start_t_search:])
        idx_t = start_t_search + idx_t_relative
        v_t = arr[idx_t]
        
        # Calcolo dell'angolo 3D
        dot_product = np.dot(v_qrs, v_t)
        norm_product = np.linalg.norm(v_qrs) * np.linalg.norm(v_t)
        cos_theta = np.clip(dot_product / norm_product, -1.0, 1.0)
        
        return float(np.degrees(np.arccos(cos_theta)))

    def compute_qrs_t_angle_error(self) -> tuple:
        """
        Calcola l'angolo QRS-T per entrambi i VCG e ne estrae l'errore.
        Returns:
            tuple: (angolo_ref, angolo_test, errore_angolare_assoluto)
        """
        angle_ref = self._compute_qrs_t_angle_for_array(self.ref_array)
        angle_test = self._compute_qrs_t_angle_for_array(self.test_array)
        error = abs(angle_ref - angle_test)
        
        return angle_ref, angle_test, error

    def compute_ae(self) -> float:
        """Angular Error (AE) globale."""
        p_peak = self._get_peak_vector(self.test_array)
        q_peak = self._get_peak_vector(self.ref_array)
        
        p_hat = p_peak / (np.linalg.norm(p_peak) + 1e-8)
        q_hat = q_peak / (np.linalg.norm(q_peak) + 1e-8)
        
        dot_product = np.clip(np.dot(p_hat, q_hat), -1.0, 1.0)
        return float(np.degrees(np.arccos(dot_product)))

    def compute_pve(self) -> float:
        """Peak Vector Error (PVE) normalizzato."""
        p_peak = self._get_peak_vector(self.test_array)
        q_peak = self._get_peak_vector(self.ref_array)
        
        diff_norm = np.linalg.norm(p_peak - q_peak)
        q_norm = np.linalg.norm(q_peak) + 1e-8
        return float(diff_norm / q_norm)

    def compute_lar_global(self) -> dict:
        """Loop Area Ratio (LAR) sui tre piani."""
        if len(self.ref_array) < 3 or len(self.test_array) < 3:
            return {'xy': np.nan, 'xz': np.nan, 'yz': np.nan}

        def shoelace_area(pts):
            x, y = pts[:, 0], pts[:, 1]
            return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

        results = {}
        planes = {'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)}
        for plane_name, (i, j) in planes.items():
            area_ref = shoelace_area(self.ref_array[:, [i, j]])
            area_test = shoelace_area(self.test_array[:, [i, j]])
            results[plane_name] = float(area_test / (area_ref + 1e-8))
        return results

    def compute_ar_global(self) -> dict:
        """Aspect Ratio (AR) 3D globale."""
        def _calc_single_ar(trajectory):
            if len(trajectory) < 3: return np.nan
            cov_matrix = np.cov(trajectory.T)
            eigenvalues = np.sort(np.abs(np.linalg.eigvals(cov_matrix)))[::-1]
            return float(eigenvalues[0] / (eigenvalues[1] + 1e-8))

        ar_ref = _calc_single_ar(self.ref_array)
        ar_test = _calc_single_ar(self.test_array)
        ar_diff = np.nan if (np.isnan(ar_ref) or np.isnan(ar_test)) else abs(ar_test - ar_ref)
        
        return {'AR_ref': ar_ref, 'AR_test': ar_test, 'AR_discrepancy': ar_diff}



def compute_all_metrics(result_vcgs: dict) -> pd.DataFrame:
    """Trasforma il dizionario dei VCG estratti in un DataFrame di metriche raw."""
    all_results = []
    print("\n[*] Calcolo delle metriche VCG in corso...")
    
    for patient_code, beats in result_vcgs.items():
        for beat_data in beats:
            beat_idx = beat_data['beat_idx']
            vcg_gt = beat_data['VCG_gt']
            vcg_fmm = beat_data['VCG_fmm']
            vcg_kors = beat_data['VCG_kors']
            
            def append_method_record(method_name, vcg_test):
                comp = VCGComparator(vcg_gt, vcg_test)
                lar = comp.compute_lar_global()
                ar = comp.compute_ar_global()
                
                all_results.append({
                    'patient_id': patient_code,
                    'beat_idx': beat_idx,
                    'Metodo': method_name,
                    'LAR_xy': lar['xy'],
                    'LAR_xz': lar['xz'],
                    'LAR_yz': lar['yz'],
                    'AR_discrepancy': ar['AR_discrepancy'],
                    'AE': comp.compute_ae(),
                    'PVE': comp.compute_pve(),
                    'QRST_angle': comp.compute_qrs_t_angle_error()[2]
                })

            append_method_record('FMM_Allineato', vcg_fmm)
            append_method_record('Kors', vcg_kors)

    return pd.DataFrame(all_results)


def create_aggregated_metrics_dataframe(df_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggrega le metriche raw calcolando Media e Dev.Std per paziente e metodo."""
    metrics_to_aggregate = ['LAR_xy', 'LAR_xz', 'LAR_yz', 'AR_discrepancy', 'AE', 'PVE', 'QRST_angle']
    cols_to_agg = [col for col in metrics_to_aggregate if col in df_metrics.columns]
    
    agg_funcs = {col: ['mean', 'std'] for col in cols_to_agg}
    df_agg = df_metrics.groupby(['patient_id', 'Metodo']).agg(agg_funcs)
    
    df_agg.columns = [f"{col[0]}_{col[1]}" for col in df_agg.columns]
    df_agg = df_agg.reset_index()
    
    beat_counts = df_metrics.groupby(['patient_id', 'Metodo'])['beat_idx'].count().reset_index()
    beat_counts = beat_counts.rename(columns={'beat_idx': 'num_beats_analyzed'})
    
    return pd.merge(df_agg, beat_counts, on=['patient_id', 'Metodo'])