import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import config
from data_loader import get_df_patient, get_R_preproc
from vcg.visualizer import VCGVisualizer

def split_patients(path_to_R: str):
    patient_codes = [filename[:-12] for filename in sorted(os.listdir(path_to_R))]
    
    train_patients, test_patients = train_test_split(
        patient_codes,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        shuffle=config.SHUFFLE
    )
    return train_patients, test_patients

# ESTRAZIONE VCG GROUND TRUTH

def extract_training_gt_vcgs(train_patients, limit=None) -> dict:
    """
    Estrae i VCG di Ground Truth per i pazienti di training.
    """
    if limit is None:
        limit = len(train_patients)

    print(f"[*] Inizio estrazione VCG Ground Truth (Limite pazienti: {limit})...")
    gt_vcg_dict = {}

    for j, patient_code in enumerate(train_patients):
        if j >= limit:  
            break
            
        print(f"    -> Elaborazione paziente: {patient_code}")
        
        patient_df = get_df_patient(patient_code)
    
        vcg_beats = []

        for beat_idx in range(config.NUM_BEATS):
            try:
                beat_start_idx, rpeak, beat_end_idx, m_detrend = get_R_preproc(patient_code, beat_idx)
                
                patient_gt_coords = patient_df[['vx', 'vy', 'vz']]
                VCG_gt_raw = {
                    "X": np.array(patient_gt_coords["vx"][beat_start_idx:beat_end_idx]), 
                    "Y": np.array(patient_gt_coords["vy"][beat_start_idx:beat_end_idx]), 
                    "Z": np.array(patient_gt_coords["vz"][beat_start_idx:beat_end_idx])
                }

                vcg_beats.append(VCG_gt_raw) 
                
            except Exception as e:
                print(f"[!] Errore nell'estrazione del battito {beat_idx} per {patient_code}: {e}")
                continue
        
        if vcg_beats:
            gt_vcg_dict[patient_code] = vcg_beats

    print("[*] Estrazione completata.\n")
    return gt_vcg_dict

# PCA E ALLINEAMENTO

def extract_main_axis(vcg_matrix: np.ndarray) -> np.ndarray:
    """Estrae il primo autovettore (direzione principale) da una matrice VCG (L x 3)."""
    pca = PCA(n_components=1)
    pca.fit(vcg_matrix)
    return pca.components_[0]

def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calcola l'angolo in gradi tra due vettori 3D."""
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)
    
    angle_deg = np.degrees(angle_rad)
    if angle_deg > 90:
        angle_deg = 180 - angle_deg
        
    return angle_deg

def format_patient_beats_to_matrix(beats_list: list) -> np.ndarray:
    """Formatta la lista dei battiti in una singola matrice N x 3."""
    all_points = []
    for beat_dict in beats_list:
        beat_matrix = np.column_stack((beat_dict["X"], beat_dict["Y"], beat_dict["Z"]))
        all_points.append(beat_matrix)
    return np.vstack(all_points)

def compute_inter_patient_variance(patient_vcg_dict: dict) -> list:
    """Calcola le statistiche di disallineamento spaziale tra tutti i pazienti."""
    axes = {}

    for patient_id, beats_list in patient_vcg_dict.items():
        vcg_matrix = format_patient_beats_to_matrix(beats_list)
        axes[patient_id] = extract_main_axis(vcg_matrix)
        
    angles = []
    
    for p1, p2 in itertools.combinations(axes.keys(), 2):
        ang = angle_between_vectors(axes[p1], axes[p2])
        angles.append(ang)
        
    if angles:
        mean_angle = np.mean(angles)
        std_angle = np.std(angles)
        max_angle = np.max(angles)
        
        print(f"[*] Analisi PCA di Allineamento Ground Truth completata:")
        print(f"    - Pazienti analizzati: {len(axes)}")
        print(f"    - Angolo medio di disallineamento: {mean_angle:.2f}° ± {std_angle:.2f}°")
        print(f"    - Disallineamento massimo trovato: {max_angle:.2f}°\n")
    else:
        print("[!] Dati insufficienti per calcolare la varianza (servono almeno 2 pazienti).")
        
    return angles

# VISUALIZZAZIONE

def save_patient_vcg_3d(patient_id: str, beats_list: list, save_dir: str):

    vcg_dict_format = {i: beat for i, beat in enumerate(beats_list)}
    
    labels = [f"Battito {i}" for i in range(len(beats_list))]
    
    visualizer = VCGVisualizer(vcg_waves=vcg_dict_format, labels=labels)
    
    fig = visualizer.plot_3d(show=False)
    
    ax = fig.gca()
    ax.set_title(f"VCG 3D GT - Paziente {patient_id}", fontsize=14)
    
    if len(beats_list) > 10 and ax.get_legend():
        ax.get_legend().remove()
    
    filepath = os.path.join(save_dir, f"{patient_id}.png")
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    
    plt.close(fig)

def plot_patient_electrical_axes_elegant(patient_vcg_dict: dict, save_path: str = None):

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.grid(False)
    
    lim = 1.1
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])
    
    label_font_size = 12
    tick_font_size = 9
    pad_distance = 15 
    
    ax.set_xlabel('Asse X (Frank)', fontsize=label_font_size, labelpad=pad_distance)
    ax.set_ylabel('Asse Y (Frank)', fontsize=label_font_size, labelpad=pad_distance)
    ax.set_zlabel('Asse Z (Frank)', fontsize=label_font_size, labelpad=pad_distance)
    
    ax.tick_params(axis='x', labelsize=tick_font_size, pad=5)
    ax.tick_params(axis='y', labelsize=tick_font_size, pad=5)
    ax.tick_params(axis='z', labelsize=tick_font_size, pad=5)
    
    ax.set_title('Dispersione Spaziale degli Assi Elettrici Principali (PC1)\nper Paziente (Ground Truth)', fontsize=15, pad=20)
    
    ax.plot([-lim, lim], [0, 0], [0, 0], color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax.plot([0, 0], [-lim, lim], [0, 0], color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax.plot([0, 0], [0, 0], [-lim, lim], color='gray', linestyle='--', alpha=0.4, linewidth=1)
    
    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color='#cccccc', alpha=0.1, linewidth=0.5)

    cmap = plt.get_cmap('tab20', len(patient_vcg_dict))
    
    for idx, (patient_id, beats_list) in enumerate(patient_vcg_dict.items()):
        vcg_matrix = format_patient_beats_to_matrix(beats_list)
        vector = extract_main_axis(vcg_matrix)
        
        if vector[1] < 0: 
            vector = -vector
            
        ax.quiver(0, 0, 0, vector[0], vector[1], vector[2], 
                  color=cmap(idx), label=patient_id, 
                  arrow_length_ratio=0.1, linewidth=2)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=8, frameon=False)
              
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        

def main():
    train_patients, test_patients = split_patients(config.R_PARAMS_DIR)
    print("=== Analisi Allineamento Ground Truth ===")

    output_dir = os.path.join("../../results", "test_allineamento_gt_train")
    os.makedirs(output_dir, exist_ok=True)
    
    gt_vcg_dict = extract_training_gt_vcgs(train_patients, limit=None)
    
    if not gt_vcg_dict:
        print("[!] Nessun dato estratto. Uscita dallo script.")
        return

    print("\n[*] Generazione e salvataggio dei plot 3D individuali in corso...")
    for patient_id, beats_list in gt_vcg_dict.items():
        save_patient_vcg_3d(patient_id, beats_list, save_dir=output_dir)
    print(f"    -> Salvati {len(gt_vcg_dict)} grafici individuali.")

    angles = compute_inter_patient_variance(gt_vcg_dict)
    
    plot_patient_electrical_axes_elegant(gt_vcg_dict, save_path=os.path.join(output_dir, "allineamento_pca_gt.png"))
    
if __name__ == "__main__":
    main()