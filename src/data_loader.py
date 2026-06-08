import numpy as np
from utils import get_R_preproc, get_R_params, get_df_patient, calculate_kors_vcg
from vcg.calculator import VCGCalculator


def extract_all_vcgs_for_beat(patient_code, beat_idx, isoelectric_samples, percentile):
    """
    Estrae e normalizza VCG FMM, GT e Kors per un dato battito.
    Ritorna: (VCG_fmm, VCG_gt, VCG_kors). Se fallisce, ritorna (None, None, None).
    """
    try:
        # Estrazione base
        beat_start_idx, rpeak, beat_end_idx, m_detrend = get_R_preproc(patient_code, beat_idx)
        n_obs = beat_end_idx - beat_start_idx
        
        patient_df = get_df_patient(patient_code)
        patient_gt_coords = patient_df[["vx", "vy", "vz"]]
        
        # GT Raw
        VCG_gt_raw = {
            "X": np.array(patient_gt_coords["vx"][beat_start_idx:beat_end_idx]), 
            "Y": np.array(patient_gt_coords["vy"][beat_start_idx:beat_end_idx]), 
            "Z": np.array(patient_gt_coords["vz"][beat_start_idx:beat_end_idx])
        }

        # Parametri e FMM
        params_per_lead = get_R_params(patient_code, beat_idx)
        time_points = np.linspace(2 * np.pi / n_obs, 2 * np.pi, n_obs)
        vcg_calculator = VCGCalculator(params_per_lead[1], params_per_lead[3], time_points)
        VCG_fmm_raw = vcg_calculator.calculate_total_vcg(include_baseline=False)

        # Kors
        VCG_kors_raw = calculate_kors_vcg({
            'I': m_detrend[0][beat_start_idx:beat_end_idx],
            'II': m_detrend[1][beat_start_idx:beat_end_idx],
            'V1': m_detrend[6][beat_start_idx:beat_end_idx],
            'V2': m_detrend[7][beat_start_idx:beat_end_idx],
            'V3': m_detrend[8][beat_start_idx:beat_end_idx],
            'V4': m_detrend[9][beat_start_idx:beat_end_idx],
            'V5': m_detrend[10][beat_start_idx:beat_end_idx],
            'V6': m_detrend[11][beat_start_idx:beat_end_idx]
        })

        # Normalizzazione
        VCG_fmm = vcg_calculator.center_and_normalize_vcg(VCG_fmm_raw, isoelectric_samples, percentile)
        VCG_gt = vcg_calculator.center_and_normalize_vcg(VCG_gt_raw, isoelectric_samples, percentile)
        VCG_kors = vcg_calculator.center_and_normalize_vcg(VCG_kors_raw, isoelectric_samples, percentile)

        return VCG_fmm, VCG_gt, VCG_kors

    except Exception as e:
        # print(f"[!] Errore nel battito {beat_idx} - paziente {patient_code}: {e}")
        return None, None, None