import numpy as np
from typing import List, Dict, Tuple

class FMMAssigner:
    """
    Handles the I-Step (Wave Assignment) of the FMM optimization process.
    Maps identified FMM waves to specific ECG components (P, Q, R, S, T)
    based on physiological and temporal characteristics.
    """
    
    def __init__(self):
        # Noisy Set
        self.NOISY_ALPHA = (np.pi - 0.25, np.pi + 0.25)
        self.NOISY_MIN_OMEGA = 0.008
        self.NOISY_MAX_OMEGA = 0.5
        self.MIN_EXTRA_VAR = 0.15
        
        # R Wave
        self.R_LEFT_DIST_ALPHA_N = [0.15, 0.2, 0.3]
        self.R_RIGHT_DIST_ALPHA_N = 0.2
        self.R_MAX_OMEGA = [0.25, 0.5]
        self.R_MIN_VAR_MAX_OMEGA = 0.25
        
        # R Confusions (from original R code thresholds)
        self.LOW_VAR_R_P = 0.15
        self.LOW_VAR_R_Q = [0.15, 0.25]
        self.MIN_VAR_Q_TO_R = 0.12
        self.LOW_VAR_R_S = 0.25
        self.HIGH_VAR_Q = 0.15
        self.HIGH_VAR_S = 0.15
        self.MIN_VAR_U_TO_R = 0.05
        
        # P Wave
        self.P_MAX_OMEGA = [0.35, 0.40]
        self.P_MIN_VAR_MAX_OMEGA = 0.10
        self.MIN_DIST_PR = [0.065, 0.25]
        self.MAX_DIST_PR = [1.5, 1.8]
        self.MIN_DIST_PN = 0.065
        
        # Q Wave
        self.Q_MAX_OMEGA = [0.25, 0.30]
        self.Q_MIN_VAR_MAX_OMEGA = 0.15
        self.MAX_DIST_QR = [0.25, 0.40]
        
        # S Wave
        self.S_MAX_OMEGA = [0.30, 0.40]
        self.S_MIN_VAR_MAX_OMEGA = 0.25
        self.MAX_DIST_RS = 0.5
        self.MAX_DIST_SN = 0.5
        self.OMEGA_S_OR_T = 0.08
        
        # T Wave
        self.MIN_DIST_RT = [0.30, 0.40]
        self.MIN_DIST_NT = 0.3

    def is_alpha_left(self, alpha1: float, alpha2: float) -> bool:
        """Circular comparison for temporal location alpha."""
        if np.isnan(alpha1) or np.isnan(alpha2):
            return True
        return (alpha2 > np.pi and (alpha1 < alpha2 and alpha1 > np.pi)) or \
               (alpha2 < np.pi and (alpha1 > np.pi or alpha1 < alpha2))

    def get_peaks_params(self, alpha: float, beta: float, omega: float) -> Tuple[float, float]:
        """Calculates fiducial peak locations (Upper and Lower)."""
        peak_u = (alpha + 2 * np.arctan2(1/omega * np.sin(-beta/2), np.cos(-beta/2))) % (2 * np.pi)
        peak_l = (alpha + 2 * np.arctan2(1/omega * np.sin((np.pi - beta)/2), np.cos((np.pi - beta)/2))) % (2 * np.pi)
        return peak_u, peak_l

    def get_noisy_set(self, alpha_arr: np.ndarray, omega_arr: np.ndarray) -> np.ndarray:
        """Identifies components that likely represent noise."""
        return ((self.NOISY_ALPHA[0] < alpha_arr) & (alpha_arr < self.NOISY_ALPHA[1])) | \
               (omega_arr > self.NOISY_MAX_OMEGA) | (omega_arr < self.NOISY_MIN_OMEGA)

    def get_free_set(self, n_waves: int, unused_comp: List[int] = None, free_waves: List[int] = None) -> np.ndarray:
        """Identifies components that haven't been assigned yet."""
        mask = np.ones(n_waves, dtype=bool)
        if unused_comp:
            for c in unused_comp:
                mask[c] = False
        if free_waves is not None:
            temp_mask = np.zeros(n_waves, dtype=bool)
            temp_mask[free_waves] = True
            mask = mask & temp_mask
        return mask

    def assign_r_wave(self, alpha_arr: np.ndarray, omega_arr: np.ndarray, 
                    relevant_var: np.ndarray, alpha_n: float, 
                    unused_comp: List[int], beta_matrix: np.ndarray) -> int:
        """Logic for identifying the R wave among fitted components."""
        noisy_mask = self.get_noisy_set(alpha_arr, omega_arr)
        free_mask = self.get_free_set(len(alpha_arr), unused_comp)
        
        distances_to_alpha_n = 1 - np.cos(alpha_arr - alpha_n)
        left_waves = np.array([self.is_alpha_left(a, alpha_n) for a in alpha_arr])
        
        # SuperSetR: Top 3 variance waves, with omega bounds and distance bounds
        top3_idx = np.argsort(relevant_var)[-3:]
        var_cond = np.zeros_like(relevant_var, dtype=bool)
        var_cond[top3_idx] = True
        
        omega_var_cond = (omega_arr < self.R_MAX_OMEGA[0]) | \
                        ((omega_arr < self.R_MAX_OMEGA[1]) & (relevant_var > self.R_MIN_VAR_MAX_OMEGA))
        
        dist_cond = (left_waves & (distances_to_alpha_n < self.R_LEFT_DIST_ALPHA_N[2])) | \
                    (~left_waves & (distances_to_alpha_n < self.R_RIGHT_DIST_ALPHA_N))
        
        super_set_r = (~noisy_mask) & free_mask & var_cond & omega_var_cond & dist_cond
        
        if np.any(super_set_r):
            candidates = np.where(super_set_r)[0]
            return candidates[np.argmax(relevant_var[candidates])]
        
        return np.argmax(relevant_var)

    def assign_waves(self, parameters: List[Dict], n_obs: int, annotation: float) -> List[Dict]:
        """
        I-Step: Maps identified FMM waves to specific ECG components (P, Q, R, S, T).
        """
        n_waves = len(parameters[0]["Alpha"])
        alpha_arr = np.array(parameters[0]["Alpha"])
        omega_arr = np.array(parameters[0]["Omega"])
        
        var_matrix = np.array([p["Var"] for p in parameters])
        relevant_var = np.mean(var_matrix, axis=0)
        beta_matrix = np.array([p["Beta"] for p in parameters])
        
        alpha_n = (annotation / n_obs) * 2 * np.pi
        
        unused_comp = []
        r_idx = self.assign_r_wave(alpha_arr, omega_arr, relevant_var, alpha_n, unused_comp, beta_matrix)
        
        labels = [f"X{i}" for i in range(n_waves)]
        labels[r_idx] = "R"
        
        alpha_r = alpha_arr[r_idx]
        left_mask = np.array([self.is_alpha_left(a, alpha_r) for a in alpha_arr])
        left_mask[r_idx] = False
        right_mask = ~left_mask
        right_mask[r_idx] = False
        
        left_indices = np.where(left_mask)[0]
        if len(left_indices) > 0:
            left_alphas = alpha_arr[left_indices]
            dists = (alpha_r - left_alphas + 2*np.pi) % (2*np.pi)
            sorted_left = left_indices[np.argsort(dists)]
            labels[sorted_left[0]] = "Q"
            if len(sorted_left) > 1:
                labels[sorted_left[1]] = "P"
                
        right_indices = np.where(right_mask)[0]
        if len(right_indices) > 0:
            right_alphas = alpha_arr[right_indices]
            dists = (right_alphas - alpha_r + 2*np.pi) % (2*np.pi)
            sorted_right = right_indices[np.argsort(dists)]
            labels[sorted_right[0]] = "S"
            if len(sorted_right) > 1:
                labels[sorted_right[1]] = "T"

        target_order = ["P", "Q", "R", "S", "T"]
        label_to_idx = {label: i for i, label in enumerate(labels)}
        
        new_parameters = []
        for lead_res in parameters:
            new_res = {key: [] for key in lead_res.keys()}
            for target_label in target_order:
                if target_label in label_to_idx:
                    idx = label_to_idx[target_label]
                    for key in lead_res.keys():
                        new_res[key].append(lead_res[key][idx])
                else:
                    for key in lead_res.keys():
                        new_res[key].append(np.nan)
            new_parameters.append(new_res)
            
        return new_parameters
