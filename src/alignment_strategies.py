import numpy as np
from abc import ABC, abstractmethod
from utils import compute_affine_transform, apply_transform

class AlignmentStrategy(ABC):
    """Interfaccia base per le strategie di allineamento."""
    
    @abstractmethod
    def fit(self, train_data_extractor, train_patients, num_beats):
        """Metodo per calcolare la matrice globale (se necessario)."""
        pass

    @abstractmethod
    def align(self, vcg_fmm, vcg_gt):
        """Metodo per allineare il singolo VCG FMM al Ground Truth."""
        pass

class LocalAlignmentStrategy(AlignmentStrategy):
    """Allineamento Locale: overfitting battito per battito."""
    
    def fit(self, train_data_extractor, train_patients, num_beats):
        # Il metodo locale non ha bisogno di training globale.
        print("[*] Strategia Locale selezionata: Nessun training globale richiesto.")
        pass

    def align(self, vcg_fmm, vcg_gt):
        T_local = compute_affine_transform(source_vcg=vcg_fmm, target_vcg=vcg_gt)
        return apply_transform(vcg_fmm, T_local)


class GlobalLstsqStrategy(AlignmentStrategy):
    """Matrice Globale con metodo minimi quadrati."""
    
    def __init__(self):
        self.T_global = None

    def fit(self, train_data_extractor, train_patients, num_beats):
        print("[*] Training Strategia Globale (Minimi Quadrati)...")
        fmm_points, gt_points = [], []
        
        for patient_code in train_patients:
            for beat_idx in range(num_beats):
                vcg_fmm, vcg_gt, _ = train_data_extractor(patient_code, beat_idx)
                if vcg_fmm is not None and vcg_gt is not None:
                    fmm_points.append(np.column_stack((vcg_fmm['X'], vcg_fmm['Y'], vcg_fmm['Z'])))
                    gt_points.append(np.column_stack((vcg_gt['X'], vcg_gt['Y'], vcg_gt['Z'])))
        
        self.T_global = np.linalg.lstsq(np.vstack(fmm_points), np.vstack(gt_points), rcond=None)[0]
        print("    -> Matrice T_global calcolata con successo.")

    def align(self, vcg_fmm, vcg_gt):
        if self.T_global is None:
            raise ValueError("La strategia globale deve chiamare fit() prima di align().")
        return apply_transform(vcg_fmm, self.T_global)


class GlobalAvgStrategy(AlignmentStrategy):
    """Matrice Globale come media delle matrici locali."""
    
    def __init__(self):
        self.T_global = None

    def fit(self, train_data_extractor, train_patients, num_beats):
        print("[*] Training Strategia Globale (Media Matrici Locali)...")
        T_locals = []
        
        for patient_code in train_patients:
            for beat_idx in range(num_beats):
                vcg_fmm, vcg_gt, _ = train_data_extractor(patient_code, beat_idx)
                if vcg_fmm is not None and vcg_gt is not None:
                    T_locals.append(compute_affine_transform(source_vcg=vcg_fmm, target_vcg=vcg_gt))
        
        self.T_global = np.mean(T_locals, axis=0)
        print("    -> Matrice T_global (Media) calcolata con successo.")

    def align(self, vcg_fmm, vcg_gt):
        if self.T_global is None:
            raise ValueError("La strategia globale deve chiamare fit() prima di align().")
        return apply_transform(vcg_fmm, self.T_global)