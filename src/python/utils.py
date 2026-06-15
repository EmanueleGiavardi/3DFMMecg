import numpy as np
import pandas as pd
import json
import os


def calculate_kors_vcg(reconstructed_leads_dict: dict) -> dict:
    """
    Converte le 8 derivazioni ECG standard (I, II, V1-V6) nel VCG di Frank (X, Y, Z) 
    utilizzando la matrice di trasformazione di Kors.
    
    Args:
        reconstructed_leads_dict: Dizionario con le chiavi 'I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'
                                  Ognuna contiene l'array numpy del segnale FMM ricostruito (es. 797 campioni)
                                  
    Returns:
        dict: Dizionario con le chiavi 'X', 'Y', 'Z' del VCG di Frank stimato.
    """

    # righe: X, Y, Z. 
    # colonne : I, II, V1, V2, V3, V4, V5, V6
    kors_matrix = np.array([
        [ 0.38, -0.07, -0.13,  0.05, -0.01,  0.14,  0.06,  0.54], 
        [-0.07,  0.93,  0.06, -0.02, -0.05,  0.06, -0.17,  0.13], 
        [ 0.11, -0.23, -0.43, -0.06, -0.14, -0.20, -0.11,  0.31]
    ])
    
    leads_matrix = np.array([
        reconstructed_leads_dict['I'],
        reconstructed_leads_dict['II'],
        reconstructed_leads_dict['V1'],
        reconstructed_leads_dict['V2'],
        reconstructed_leads_dict['V3'],
        reconstructed_leads_dict['V4'],
        reconstructed_leads_dict['V5'],
        reconstructed_leads_dict['V6']
    ])
    
    result = np.dot(kors_matrix, leads_matrix)
    
    return {
        'X': result[0, :],
        'Y': result[1, :],
        'Z': result[2, :]
    }

def compute_affine_transform(source_vcg: dict, target_vcg: dict) -> np.ndarray:
    """
    Calcola la matrice di trasformazione affine 3D tramite minimi quadrati
    
    Args:
        source_vcg (dict): VCG sorgente (VCG FMM)
        target_vcg (dict): VCG target (VCG ground truth)
        
    Returns:
        np.ndarray: Matrice di trasformazione (3, 3)
    """

    A = np.column_stack((source_vcg['X'], source_vcg['Y'], source_vcg['Z']))
    B = np.column_stack((target_vcg['X'], target_vcg['Y'], target_vcg['Z']))
    
    T, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)
    
    return T

def apply_transform(source_vcg: dict, T: np.ndarray) -> dict:
    """
    Applica la matrice di trasformazione al VCG sorgente.
    
    Args:
        source_vcg (dict): VCG da trasformare
        T (np.ndarray): Matrice di trasformazione (3, 3)
        
    Returns:
        dict: VCG mappato
    """
    A = np.column_stack((source_vcg['X'], source_vcg['Y'], source_vcg['Z']))
    mapped_vcg = np.dot(A, T)
    
    return {
        'X': mapped_vcg[:, 0],
        'Y': mapped_vcg[:, 1],
        'Z': mapped_vcg[:, 2]
    }


# estrazione parametri R da json

def get_R_params(patient_code, beat_idx, params_dir:str):
    with open(f'{params_dir}/{patient_code}_PARAMS.json', 'r') as f: r_params = json.load(f)
    r_params_beat = r_params[beat_idx]
    lead_names_r = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]
    wave_order = ['P', 'Q', 'R', 'S', 'T']
    params_per_lead_R = []

    for lead in lead_names_r:
        lead_data = r_params_beat[lead]
        waves_dict = {w['_row']: w for w in lead_data}
        
        target_format = {'M': [], 'A': [], 'Alpha': [], 'Beta': [], 'Omega': [], 'Var': []}
        for wave in wave_order:
            try: 
                data = waves_dict[wave]
                for param in target_format.keys(): target_format[param].append(np.float64(data[param]))
            except:
                raise(f"Formato parametri non valido per battito {beat_idx}")
                
        params_per_lead_R.append(target_format)

    return params_per_lead_R

def get_R_preproc(patient_code, beat_idx, preproc_dir:str):
    with open(f'{preproc_dir}/{patient_code}_PREPROC.json', 'r') as f: r_preproc = json.load(f)
    m_detrend = np.asarray(r_preproc['m_detrend'], dtype=np.float64) 

    start = r_preproc['beats'][beat_idx]["inizio"] - 1
    rpeak = r_preproc['beats'][beat_idx]["picco_r"] - 1
    end = r_preproc['beats'][beat_idx]["fine"] - 1
    
    return start, rpeak, end, m_detrend.T

def get_df_patient(patient_code, dataset_dir:str):
    return pd.read_csv(os.path.join(dataset_dir, patient_code+'.csv'))