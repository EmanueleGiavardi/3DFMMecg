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

    def calculate_total_vcg(self, include_baseline: bool = True) -> dict:
        """
        Calcola la traiettoria del VCG completo eseguendo la somma vettoriale 
        punto per punto di tutte le onde componenti (P, Q, R, S, T) e aggiungendo 
        opzionalmente l'offset della linea di base (M).
        
        Args:
            include_baseline (bool): Se True, aggiunge l'offset isoelettrico globale 
                                     ricavato dai parametri M del fit.
                                     
        Returns:
            dict: Dizionario contenente i vettori 1D delle coordinate dell'ECG completo:
                  {'X': array, 'Y': array, 'Z': array}
        """
        vcg_waves = self.calculate_axes()
        total_X = np.zeros_like(self.time_points)
        total_Y = np.zeros_like(self.time_points)
        total_Z = np.zeros_like(self.time_points)
        
        # somma vettoriale punto per punto
        for wave_data in vcg_waves.values():
            total_X += wave_data['X']
            total_Y += wave_data['Y']
            total_Z += wave_data['Z']

        if include_baseline:
            # Estraiamo l'intercetta M (usiamo il primo indice poiché M è identica 
            # per tutte le onde dello stesso lead dopo il ricalcolo congiunto)
            M_II = self.params_lead_II['M'][0] if len(self.params_lead_II.get('M', [])) > 0 else 0.0
            M_V2 = self.params_lead_V2['M'][0] if len(self.params_lead_V2.get('M', [])) > 0 else 0.0
            
            # Applichiamo alle costanti M le stesse proiezioni lineari degli assi:
            # Asse X dipendente da Lead II
            total_X += M_II 
            
            # Asse Y rappresenta la componente immaginaria/Hilbert (la componente continua DC è nulla)
            total_Y += 0.0 
            
            # Asse Z modulato su V2 e corretto per l'apporto di Y
            total_Z += (M_V2 - 2 * 0.0) 
            
        return {
            'X': total_X,
            'Y': total_Y,
            'Z': total_Z
        }


    def center_and_normalize_vcg(self, vcg_dict: dict, isoelectric_samples: int = 20, percentile: int = 99) -> np.ndarray:
        """
        Centra il VCG sull'origine (0,0,0) usando il tratto isoelettrico (i primi isoelectric_samples campioni del segnale) e 
        lo normalizza spazialmente usando la norma al percentile specificato.
        
        Args:
            vcg_array: Array numpy di shape (N_campioni, 3) contenente X, Y, Z.
            isoelectric_samples: Numero di campioni all'inizio del battito da usare 
                                per stimare il vero punto di riposo del cuore.
                                
        Returns:
            dict: VCG centrato e normalizzato (shape N_campioni, 3).
        """

        vcg_array = np.column_stack((vcg_dict['X'], vcg_dict['Y'], vcg_dict['Z']))

        if isoelectric_samples != 0:
            baseline_offset = np.mean(vcg_array[:isoelectric_samples, :], axis=0)
            centered_vcg = vcg_array - baseline_offset
        else:
            centered_vcg = vcg_array
        
        norms = np.linalg.norm(centered_vcg, axis=1)
        p_norm = np.percentile(norms, percentile)
        
        if p_norm < 1e-10: return vcg_dict

        normalized_vcg = centered_vcg / p_norm
        
        return {
            'X': normalized_vcg[:, 0],
            'Y': normalized_vcg[:, 1],
            'Z': normalized_vcg[:, 2]
        }
        
        