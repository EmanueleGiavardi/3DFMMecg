import numpy as np
import pandas as pd

class VCGComparator:
    """Classe dedicata al calcolo delle metriche di picco e morfologiche."""
    
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
        Ritorna: tupla: (angolo_ref, angolo_test, errore_angolare_assoluto)
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
        """
        Loop Area Ratio (LAR) sui tre piani.
        Ritorna: dict: LAR sui tre piani (xy, xz, yz)
        """
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
        """Aspect Ratio (AR) 3D globale.
        Ritorna: dict: AR del ground truth, AR del test e discrepancy
        """
        def _calc_single_ar(trajectory):
            if len(trajectory) < 3: return np.nan
            cov_matrix = np.cov(trajectory.T)
            eigenvalues = np.sort(np.abs(np.linalg.eigvals(cov_matrix)))[::-1]
            return float(eigenvalues[0] / (eigenvalues[1] + 1e-8))

        ar_ref = _calc_single_ar(self.ref_array)
        ar_test = _calc_single_ar(self.test_array)
        ar_diff = np.nan if (np.isnan(ar_ref) or np.isnan(ar_test)) else abs(ar_test - ar_ref)
        
        return {'AR_ref': ar_ref, 'AR_test': ar_test, 'AR_discrepancy': ar_diff}


def slice_vcg_dict(vcg: dict, start_idx: int, end_idx: int) -> dict:
    """Segmenta un dizionario VCG in un intervallo temporale specifico."""
    return {
        'X': np.array(vcg['X'])[start_idx:end_idx],
        'Y': np.array(vcg['Y'])[start_idx:end_idx],
        'Z': np.array(vcg['Z'])[start_idx:end_idx]
    }

def compute_all_metrics(result_vcgs: dict) -> pd.DataFrame:
    """Trasforma il dizionario dei VCG estratti in un DataFrame di metriche per singolo complesso."""
    all_results = []
    print("\n[*] Calcolo delle metriche VCG per complessi (P, QRS, T) in corso...")
    
    for patient_code, beats in result_vcgs.items():
        for beat_data in beats:
            beat_idx = beat_data['beat_idx']
            vcg_gt = beat_data['VCG_gt']
            vcg_fmm = beat_data.get('VCG_fmm')
            vcg_kors = beat_data.get('VCG_kors')
            
            # trova l'indice del Picco R nel Ground Truth (massima ampiezza vettoriale 3D)
            gt_matrix = np.column_stack((vcg_gt['X'], vcg_gt['Y'], vcg_gt['Z']))
            magnitudes = np.linalg.norm(gt_matrix, axis=1)
            r_peak_idx = np.argmax(magnitudes)
            N = len(magnitudes)

            windows = {
                'P': (max(0, r_peak_idx - 250), max(0, r_peak_idx - 80)),
                'QRS': (max(0, r_peak_idx - 60), min(N, r_peak_idx + 80)),
                'T': (min(N, r_peak_idx + 120), min(N, r_peak_idx + 400))
            }
            
            def append_method_record(method_name, vcg_test):
                if vcg_test is None:
                    return
                    
                comp_global = VCGComparator(vcg_gt, vcg_test)
                
                metrics = {
                    'patient_id': patient_code,
                    'beat_idx': beat_idx,
                    'Metodo': method_name,
                    'QRST_angle': comp_global.compute_qrs_t_angle_error()[2] 
                }
                
                # itera sui 3 complessi P, QRS, T
                for complex_name, (start, end) in windows.items():
                    if end - start < 10:
                        metrics[f'AE_{complex_name}'] = np.nan
                        metrics[f'PVE_{complex_name}'] = np.nan
                        if complex_name in ['QRS', 'T']:
                            metrics[f'LAR_{complex_name}_xy'] = np.nan
                            metrics[f'LAR_{complex_name}_xz'] = np.nan
                            metrics[f'LAR_{complex_name}_yz'] = np.nan
                            metrics[f'AR_discrepancy_{complex_name}'] = np.nan
                        continue
                        
                    # segmentazione
                    vcg_gt_slice = slice_vcg_dict(vcg_gt, start, end)
                    vcg_test_slice = slice_vcg_dict(vcg_test, start, end)
                    
                    comp_slice = VCGComparator(vcg_gt_slice, vcg_test_slice)
                    
                    metrics[f'AE_{complex_name}'] = comp_slice.compute_ae()
                    metrics[f'PVE_{complex_name}'] = comp_slice.compute_pve()
                    
                    if complex_name in ['QRS', 'T']:
                        lar = comp_slice.compute_lar_global()
                        ar = comp_slice.compute_ar_global()
                        
                        metrics[f'LAR_{complex_name}_xy'] = lar['xy']
                        metrics[f'LAR_{complex_name}_xz'] = lar['xz']
                        metrics[f'LAR_{complex_name}_yz'] = lar['yz']
                        metrics[f'AR_discrepancy_{complex_name}'] = ar['AR_discrepancy']

                all_results.append(metrics)

            append_method_record('FMM_Allineato', vcg_fmm)
            append_method_record('Kors', vcg_kors)

    return pd.DataFrame(all_results)


def create_aggregated_metrics_dataframe(df_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggrega le metriche raw calcolando Media e Dev.Std per paziente e metodo."""
    
    exclude_cols = ['patient_id', 'Metodo', 'beat_idx']
    cols_to_agg = [col for col in df_metrics.columns if col not in exclude_cols]
    
    agg_funcs = {col: ['mean', 'std'] for col in cols_to_agg}
    
    df_agg = df_metrics.groupby(['patient_id', 'Metodo']).agg(agg_funcs)
    
    df_agg.columns = [f"{col[0]}_{col[1]}" for col in df_agg.columns]
    df_agg = df_agg.reset_index()
    
    beat_counts = df_metrics.groupby(['patient_id', 'Metodo'])['beat_idx'].count().reset_index()
    beat_counts = beat_counts.rename(columns={'beat_idx': 'num_beats_analyzed'})
    
    return pd.merge(df_agg, beat_counts, on=['patient_id', 'Metodo'])