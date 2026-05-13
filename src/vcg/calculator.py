import numpy as np

class VCGCalculator:
    """
    Classe per calcolare le traiettorie geometriche (assi) del Vectorcardiogramma (VCG)
    a partire dai parametri ottimizzati del modello FMM.
    """

    def __init__(self, params_lead_II: dict, params_lead_V2: dict, time_points: np.ndarray):
        """
        Inizializza il calcolatore.
        
        Args:
            params_lead_II (dict): Parametri ottimizzati FMM per derivazione DII.
            params_lead_V2 (dict): Parametri ottimizzati FMM per derivazione V2.
            time_points (np.ndarray): Array angolare del tempo del battito FMM.
        """
        self.params_lead_II = params_lead_II
        self.params_lead_V2 = params_lead_V2
        self.time_points = time_points
        self.n_waves = len(params_lead_II.get('Alpha', []))

    def _evaluate_wave_phase(self, alpha: float, beta: float, omega: float) -> np.ndarray:
        """
        Calcola la fase istantanea per l'onda FMM (equazione 1 del paper)
        phi(t) = beta + 2 * arctan(omega * tan((t - alpha)/2))
        """
        return beta + 2 * np.arctan(omega * np.tan((self.time_points - alpha) / 2))

    def calculate_axes(self) -> dict:
        """
        Costruisce la traiettoria geometrica vettoriale per tutte le componenti d'onda.
        
        Returns:
            dict: Dizionario associato ad ogni onda fittata:
                  {
                      0: {'X': [array...], 'Y': [array...], 'Z': [array...]},
                      ...
                  }
        """
        vcg_waves = {}
        
        for i in range(self.n_waves):
            A_II, A_V2 = self.params_lead_II['A'][i], self.params_lead_V2['A'][i]
            alpha_II, alpha_V2 = self.params_lead_II['Alpha'][i], self.params_lead_V2['Alpha'][i]
            beta_II, beta_V2 = self.params_lead_II['Beta'][i], self.params_lead_V2['Beta'][i]
            omega_II, omega_V2 = self.params_lead_II['Omega'][i], self.params_lead_V2['Omega'][i]

            phi_II = self._evaluate_wave_phase(alpha_II, beta_II, omega_II)
            phi_V2 = self._evaluate_wave_phase(alpha_V2, beta_V2, omega_V2)
            
            # Proiezione su asse 3D
            # 1) Asse X (Coronale Left-Right): Diretto estrapolato DII cos
            X = A_II * np.cos(phi_II)
            
            # 2) Asse Y (Coronale Superior-Inferior): Trasformata di Hilbert di DII = DII sin
            Y = A_II * np.sin(phi_II)
            
            # 3) Asse Z (Sagittale Anteroposterior): Modulato su derivazione Anteriore V2 bilanciato per Y
            Z = A_V2 * np.cos(phi_V2) - 2 * Y
            
            vcg_waves[i] = {
                'X': X,
                'Y': Y,
                'Z': Z
            }
            
        return vcg_waves
