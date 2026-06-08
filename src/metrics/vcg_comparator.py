import numpy as np

class VCGComparator:
    def __init__(self, vcg_ref: dict, vcg_test: dict):
        self.vcg_ref = vcg_ref
        self.vcg_test = vcg_test
        
        self.ref_array = np.column_stack((vcg_ref['X'], vcg_ref['Y'], vcg_ref['Z']))
        self.test_array = np.column_stack((vcg_test['X'], vcg_test['Y'], vcg_test['Z']))

    def compute_rmse(self) -> float:
        """Calcola l'RMSE (Root Mean Square Error) spaziale 3D."""
        squared_diff = (self.ref_array - self.test_array) ** 2
        squared_distances = np.sum(squared_diff, axis=1)
        return float(np.sqrt(np.mean(squared_distances)))

    def compute_spatial_correlation(self) -> float:
        """Calcola la correlazione spaziale media (Pearson) tra i due VCG."""
        corr_x = np.corrcoef(self.vcg_ref['X'], self.vcg_test['X'])[0, 1]
        corr_y = np.corrcoef(self.vcg_ref['Y'], self.vcg_test['Y'])[0, 1]
        corr_z = np.corrcoef(self.vcg_ref['Z'], self.vcg_test['Z'])[0, 1]
        return float(np.mean([corr_x, corr_y, corr_z]))

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


    def _get_peak_vector(self, trajectory: np.ndarray) -> np.ndarray:
        """Trova il vettore con la norma euclidea massima nel segmento."""
        norms = np.linalg.norm(trajectory, axis=1)
        peak_idx = np.argmax(norms)
        return trajectory[peak_idx]

    def compute_ae(self) -> float:
        """Calcola l'Angular Error (AE) in gradi tra i vettori di picco."""
        p_peak = self._get_peak_vector(self.test_array)
        q_peak = self._get_peak_vector(self.ref_array)
        
        p_hat = p_peak / (np.linalg.norm(p_peak) + 1e-8)
        q_hat = q_peak / (np.linalg.norm(q_peak) + 1e-8)
        
        return np.degrees(np.arccos(np.clip(np.dot(p_hat, q_hat), -1.0, 1.0)))

    def compute_pve(self) -> float:
        """Calcola il Peak Vector Error (PVE) normalizzato."""
        p_peak = self._get_peak_vector(self.test_array)
        q_peak = self._get_peak_vector(self.ref_array)
        
        diff_norm = np.linalg.norm(p_peak - q_peak)
        q_norm = np.linalg.norm(q_peak) + 1e-8
        
        return diff_norm / q_norm


    def compute_lar_global(self) -> dict:
            """
            Calcola il Loop Area Ratio (LAR) sui tre piani di proiezione (xy, xz, yz)
            sull'intera traccia VCG.
            """
            
            ref_seg = self.ref_array
            test_seg = self.test_array
            
            #if len(ref_seg) < 3 or len(test_seg) < 3:
            #    return {'xy': np.nan, 'xz': np.nan, 'yz': np.nan}

            def shoelace_area(pts):
                x, y = pts[:, 0], pts[:, 1]
                return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

            results = {}    
            planes = {'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)}

            for plane_name, (i, j) in planes.items():
                area_ref = shoelace_area(ref_seg[:, [i, j]])
                area_test = shoelace_area(test_seg[:, [i, j]])
                
                lar = area_test / (area_ref + 1e-8)
                results[plane_name] = lar

            return results

    def compute_ar_global(self) -> dict:
        """
        Calcola l'Aspect Ratio (AR) 3D globale sull'intera traccia VCG per la 
        traiettoria di riferimento e quella predetta, calcolandone la discrepanza.
        """
        # Utilizza direttamente l'intera traccia
        ref_seg = self.ref_array
        test_seg = self.test_array

        def _calc_single_ar(trajectory):
            # Servono almeno 3 punti per una matrice di covarianza 3x3 significativa
            if len(trajectory) < 3:
                return np.nan
                
            # np.cov richiede che le variabili (X,Y,Z) siano sulle righe, quindi si usa .T
            cov_matrix = np.cov(trajectory.T)
            
            # Calcolo autovalori e ordinamento decrescente per magnitudo
            eigenvalues = np.sort(np.abs(np.linalg.eigvals(cov_matrix)))[::-1]
            
            # Rapporto dell'asse maggiore sull'asse minore
            return eigenvalues[0] / (eigenvalues[1] + 1e-8)

        # Calcolo AR per le due traiettorie intere
        ar_ref = _calc_single_ar(ref_seg)
        ar_test = _calc_single_ar(test_seg)
        
        # Calcolo della discrepanza assoluta
        if np.isnan(ar_ref) or np.isnan(ar_test):
            ar_diff = np.nan
        else:
            ar_diff = abs(ar_test - ar_ref)

        return {
            'AR_ref': ar_ref,
            'AR_test': ar_test,
            'AR_discrepancy': ar_diff
        }
