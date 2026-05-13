import numpy as np
import scipy.optimize
from typing import List, Dict, Tuple
from optimization.utils import plot_ecg_beat
from optimization.fmm_wave import FMMWave
from optimization.fmm_assigner import FMMAssigner

class FMMOptimizer:
    """
    Classe per l'ottimizzazione dei parametri FMM per un segnale ECG multi-lead. 
    """
    
    def __init__(self, n_waves: int = 5, max_iter: int = 10, omega_max: float = 0.7):
        """
        Inizializza l'ottimizzatore FMM.
        
        Args:
            n_waves: Numero di onde da fittare (tipicamente 5 per P-Q-R-S-T).
            max_iter: Numero massimo di iterazioni del ciclo di backfitting.
            omega_max: Limite superiore per il parametro omega.
        """
        self.n_waves = n_waves
        self.max_iter = max_iter
        self.omega_max = omega_max
        self._grid = None
        self._time_points = None
        
        # Wave Assigner (I-Step)
        self.assigner = FMMAssigner()

    @staticmethod
    def generate_time_points(n_obs: int) -> np.ndarray:
        """Genera i punti temporali t in (0, 2*pi] mappati su un singolo battito."""
        return np.linspace(2 * np.pi / n_obs, 2 * np.pi, n_obs)

    @staticmethod
    def calculate_variance_explained(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcola la percentuale di varianza (Percentage of Variance nel paper) spiegata (equivalente a R2)."""
        sse = np.sum((y_true - y_pred)**2)  # somma degli errori al quadrato
        sst = np.sum((y_true - np.mean(y_true))**2) # energia del segnale di ground truth
        return 1 - (sse / sst) if sst != 0 else 0.0

    def _initialize_grid(self, n_obs: int):
        """
        Pre-calcola i blocchi di Grid Search per le coppie alpha e omega (parametri globali).
        """
        self._time_points = self.generate_time_points(n_obs)
        
        # Dimensione griglia: 48 alpha x 24 omega = 1152 combinazioni
        alpha_grid = np.linspace(0, 2 * np.pi, 48)
        omega_grid = np.exp(np.linspace(np.log(0.0001), np.log(1.0), 24))
        
        self._grid = []
        for alpha in alpha_grid:
            for omega in omega_grid:
                # Trasformazione di Moebius
                phi = 2 * np.arctan(omega * np.tan((self._time_points - alpha) / 2))
                cost = np.cos(phi)
                sint = np.sin(phi)
                
                m_matrix = np.column_stack((np.ones_like(self._time_points), cost, sint))
                # Calcolo pseudo-inversa 
                pinv_m = np.linalg.pinv(m_matrix)
                
                self._grid.append({
                    'alpha': alpha,
                    'omega': omega,
                    'pinv_m': pinv_m,
                    'cost': cost,
                    'sint': sint
                })

    def _evaluate_grid_point(self, grid_item: Dict, residuals_matrix: np.ndarray, weights: np.ndarray) -> Tuple[float, float, float]:
        """Valuta una coppia alpha e omega della griglia e ritorna l'RSS"""
        # risolve il problema di regressione lineare (avendo già a disposizione la pseudo-inversa)
        # residual matrix è il segnale di ground truth (Y_true) da fittare
        pars = grid_item['pinv_m'] @ residuals_matrix
        
        cost = grid_item['cost'][:, np.newaxis]
        sint = grid_item['sint'][:, np.newaxis]
        # Y_hat    = intercetta + omega (m1) * cost + gamma (m2) * sint
        prediction = pars[0, :] + pars[1, :] * cost + pars[2, :] * sint
        
        # Somma degli errori al quadrato pesata su tutti i leads
        rss_per_lead = np.sum((residuals_matrix - prediction)**2, axis=0)
        total_rss = rss_per_lead @ weights
        
        return grid_item['alpha'], grid_item['omega'], total_rss

    def _global_loss_function(self, params: np.ndarray, residuals_matrix: np.ndarray, weights: np.ndarray) -> float:
        """Loss function per il fine-tuning di alpha e omega usando L-BFGS-B."""
        alpha, omega = params
        
        phi = 2 * np.arctan(omega * np.tan((self._time_points - alpha) / 2))
        m_matrix = np.column_stack((np.ones_like(phi), np.cos(phi), np.sin(phi)))

        # risolve regressione lineare per fittare m_matrix su residuals_matrix (segnale di ground truth)
        # beta_hat = [intercept, m1, m2]
        beta_hat, _, _, _ = np.linalg.lstsq(m_matrix, residuals_matrix, rcond=None)
        
        # Calcolo RSS pesato
        fitted = m_matrix @ beta_hat
        rss_per_lead = np.sum((residuals_matrix - fitted)**2, axis=0)
        return rss_per_lead @ weights

    def _optimize_global_parameters(self, residuals_matrix: np.ndarray, error_weights: np.ndarray) -> Tuple[float, float]:
        """Individua i valori ottimi per i parametri globali alpha e omega"""
        best_rss = np.inf
        best_alpha, best_omega = 0.0, 0.0
        
        # 1) Viene usata una Grid Search per individuare i migliori parametri alpha e omega della griglia
        # in modo tale da partire con una buona stima per il fine-tuning
        for item in self._grid:
            # risolve il problema di regressione lineare (avendo già a disposizione la pseudo-inversa) e calcola l'errore RSS
            # legato al fitting dell'onda predetta (Y_hat / prediction) rispetto all'onda misurata (Y_true / residuals_matrix)
            alpha, omega, rss = self._evaluate_grid_point(item, residuals_matrix, error_weights)
            if rss < best_rss:
                best_rss = rss
                best_alpha, best_omega = alpha, omega
                
        # 2) Una volta trovati i migliori alpha e omega della griglia, viene effettuato un fine-tuning usando L-BFGS-B
        # (discesa del gradiente classica: ad ogni step viene valutata la _global_loss_function che restituisce l'errore
        # totale pesato su tutte le derivazioni). 
        bounds = [(-2 * np.pi, 4 * np.pi), (0.0001, self.omega_max)]
        res = scipy.optimize.minimize(
            fun=self._global_loss_function,
            x0=np.array([best_alpha, best_omega]),
            args=(residuals_matrix, error_weights),
            method='L-BFGS-B',
            bounds=bounds
        )
        
        return res.x[0], res.x[1]

    def _estimate_local_parameters(self, lead_data: np.ndarray, alpha: float, omega: float) -> Tuple[FMMWave, np.ndarray]:
        """Stima i parametri locali (M, A, Beta) di ogni singola derivazione, date le soluzioni globali ottime alpha e omega"""        
        t_star = alpha + 2 * np.arctan2(omega * np.sin((self._time_points - alpha) / 2),
                                        np.cos((self._time_points - alpha) / 2))
        
        m_matrix = np.column_stack((np.ones_like(t_star), np.cos(t_star), np.sin(t_star)))
        beta_hat, _, _, _ = np.linalg.lstsq(m_matrix, lead_data, rcond=None)
        
        m_val, delta, gamma = beta_hat
        
        fitted_values = m_val + delta * np.cos(t_star) + gamma * np.sin(t_star)
        pv = self.calculate_variance_explained(lead_data, fitted_values)
        
        amplitude = np.sqrt(delta**2 + gamma**2)
        alpha_mod = (alpha + 2 * np.pi) % (2 * np.pi)
        beta_val = (np.arctan2(-gamma, delta) + alpha_mod + 2 * np.pi) % (2 * np.pi)
        
        wave = FMMWave(
            M=m_val,
            A=amplitude,
            alpha=alpha_mod,
            beta=beta_val,
            omega=omega,
            variance_explained=pv
        )
        
        return wave, fitted_values

    def _update_error_weights(self, ecg_data: np.ndarray, fitted_waves: np.ndarray, signal_weights: np.ndarray) -> np.ndarray:
        """Aggiorna il peso di ogni derivazione in base alla varianza dei residui"""
        n_obs = ecg_data.shape[0]
        current_fit = np.sum(fitted_waves, axis=2)
        residuals = ecg_data - current_fit
        
        # Calcola la varianza dei residui (sigma) per ogni derivazione
        sigma = np.sum(residuals**2, axis=0) / (n_obs - 1)
        # Peso inverso dell'errore
        error_weights = (1 / sigma) * signal_weights
        return error_weights / np.sum(error_weights)

    def _recalculate_M_and_A(self, ecg_data: np.ndarray, lead_results: List[Dict], n_obs: int) -> List[Dict]:
        """Ricalcola M e A per ogni derivazione usando una regressione lineare multipla,
        mantenendo costanti i parametri non lineari (alpha, beta, omega)"""
        for lead_index in range(len(lead_results)):
            n_waves = len(lead_results[lead_index]["Alpha"])
            if n_waves == 0: continue
                
            lead_data = ecg_data[:, lead_index]
            
            m_matrix = np.zeros((n_obs, n_waves + 1))
            m_matrix[:, 0] = 1.0
            
            for w in range(n_waves):
                alpha = lead_results[lead_index]["Alpha"][w]
                beta = lead_results[lead_index]["Beta"][w]
                omega = lead_results[lead_index]["Omega"][w]
                
                t_star = alpha + 2 * np.arctan2(omega * np.sin((self._time_points - alpha) / 2),
                                                np.cos((self._time_points - alpha) / 2))
                
                norm_wave = np.cos(t_star + beta - alpha)
                m_matrix[:, w + 1] = norm_wave
                
            coefs, _, _, _ = np.linalg.lstsq(m_matrix, lead_data, rcond=None)
            
            recalculated_m = coefs[0]
            recalculated_as = coefs[1:]
            
            # Gestisce i valori negativi di A (potenzialmente restituiti dalla regressione) tramite phase shifting
            for w in range(n_waves):
                if recalculated_as[w] < 0:
                    recalculated_as[w] = -recalculated_as[w]
                    lead_results[lead_index]["Beta"][w] = (lead_results[lead_index]["Beta"][w] + np.pi) % (2 * np.pi)
                
                lead_results[lead_index]["M"][w] = recalculated_m
                lead_results[lead_index]["A"][w] = recalculated_as[w]
            
            # Ricalcola la varianza spiegata (PV) in modo sequenziale per ogni onda
            dm_matrix = np.zeros((n_obs, 2 * n_waves + 1))
            dm_matrix[:, 0] = 1.0
            
            cumulative_pv = []
            for w in range(n_waves):
                alpha = lead_results[lead_index]["Alpha"][w]
                omega = lead_results[lead_index]["Omega"][w]
                
                t_star = alpha + 2 * np.arctan2(omega * np.sin((self._time_points - alpha) / 2),
                                                np.cos((self._time_points - alpha) / 2))
                
                dm_matrix[:, 2*w + 1] = np.cos(t_star)
                dm_matrix[:, 2*w + 2] = np.sin(t_star)
                
                dm_subset = dm_matrix[:, :2*(w+1) + 1]
                beta_hat, _, _, _ = np.linalg.lstsq(dm_subset, lead_data, rcond=None)
                fitted_subset = dm_subset @ beta_hat
                pv = self.calculate_variance_explained(lead_data, fitted_subset)
                cumulative_pv.append(pv)
                
            var_array = [cumulative_pv[0]] + [cumulative_pv[i] - cumulative_pv[i-1] for i in range(1, n_waves)]
            
            for w in range(n_waves):
                lead_results[lead_index]["Var"][w] = var_array[w]
        return lead_results

    def _check_early_stop(self, lead_results: List[Dict]) -> bool:
        return False
    
    def fit(self, ecg_data: np.ndarray, annotation: float, plot_ecg = False) -> Tuple[List[Dict], np.ndarray]:
        """
        Fits multiple FMM waves to a multi-lead ECG segment using iterative backfitting.
        
        Args:
            ecg_data: Matrix of shape (n_observations, n_leads).
            annotation: Contextual info for wave assignment.
            
        Returns:
            A tuple (parameters_per_lead, fitted_waves_matrix).
        """

        relevant_leads_idx = [0, 1, 6, 7, 8, 9, 10, 11]
        ecg_data = ecg_data[:, relevant_leads_idx]

        n_obs, n_leads = ecg_data.shape
        self._initialize_grid(n_obs)
        
        # Initialization
        # fitted_waves è una matrice in cui, per ogni derivazione e per ogni campione, vengono memorizzati 
        # i valori delle 5 onde
        fitted_waves = np.zeros((n_obs, n_leads, self.n_waves))
        
        # Assuming relevant_leads_idx = [I, II, V1, V2, V3, V4, V5, V6]
        signal_weights = np.array([0.5, 0.5, 1/6, 1/6, 1/6, 1/6, 1/6, 1/6])
        error_weights = signal_weights / np.sum(signal_weights)

        lead_results = [{"M": [], "A": [], "Alpha": [], "Beta": [], "Omega": [], "Var": []} for _ in range(n_leads)]
        
        for iteration in range(self.max_iter):
            current_lead_results = [{"M": [], "A": [], "Alpha": [], "Beta": [], "Omega": [], "Var": []} for _ in range(n_leads)]
            for wave_to_fit in range(self.n_waves):

                # 1) CALCOLO DEI RESIDUI (ECG DATA - SOMMA DELLE ALTRE ONDE a parte quella corrente)
                # other_waves_sum è un vettore (797, 8) (cioè num_samples, num_leads) perchè questa operazione sta andando a guardare 
                # ogni quintupla di fitted_waves (cioè una lista relativa alle 5 onde da fittare, gli sta applicando una maschera 
                # in modo tale da ignorare l'onda corrente (ad esempio R), e sta sommando i valori delle altre onde 
                # (ad esempio P + Q + S + T), e questo restituisce un unico valore scalare che rappresenta l'ECG in quel punto 
                # SENZA il contributo dell'onda corrente 
                mask = np.ones(self.n_waves, dtype=bool)
                mask[wave_to_fit] = False
                other_waves_sum = np.sum(fitted_waves[:, :, mask], axis=2)
        
                # Per il calcolo dei residui reali, si sottraggono sia le oscillazioni che la baseline M calcolata al giro precedente
                baseline_m = np.array([current_lead_results[lead]["M"][0] if len(current_lead_results[lead]["M"]) > 0 else 0.0 for lead in range(n_leads)])
                residuals_matrix = ecg_data - (other_waves_sum + baseline_m)

                # 2) STIMA DEI PARAMETRI GLOBALI (ALPHA, OMEGA) validi per tutte le derivazioni 
                opt_alpha, opt_omega = self._optimize_global_parameters(residuals_matrix, error_weights)
                
                # 3) STIMA DEI PARAMETRI LOCALI (M, A, Beta) per ogni derivazione
                for lead_index in range(n_leads):
                    # wave -> onda contenente i parametri per l'onda corrente wave_to_fit alla specifica derivazione lead_index
                    # fitted_vals -> matrice contenente i valori stimati dell'onda corrente wave_to_fit alla specifica derivazione lead_index
                    wave, fitted_vals = self._estimate_local_parameters(residuals_matrix[:, lead_index], opt_alpha, opt_omega)

                    # assegna i valori dell'onda corrente alla matrice fitted_waves
                    #fitted_waves[:, lead_index, wave_to_fit] = fitted_vals - wave.M

                    current_lead_results[lead_index]["M"].append(wave.M)
                    current_lead_results[lead_index]["A"].append(wave.A)
                    current_lead_results[lead_index]["Alpha"].append(wave.alpha)
                    current_lead_results[lead_index]["Beta"].append(wave.beta)
                    current_lead_results[lead_index]["Omega"].append(wave.omega)
                    current_lead_results[lead_index]["Var"].append(wave.variance_explained)
                    
                    # stampa del processo di ottimizzazione per la prima derivazione
                    if plot_ecg and lead_index == 0:

                        lead1_min, lead1_max = np.min(ecg_data[:, 0]), np.max(ecg_data[:, 0])
                        y_margin = (lead1_max - lead1_min) * 0.1
                        visual_limits = (lead1_min - y_margin, lead1_max + y_margin)
                        current_start = residuals_matrix[:, lead_index]
                        new_residual = current_start - fitted_vals
                        plot_ecg_beat(
                            current_start, 
                            fitted_vals, 
                            new_residual,
                            title=f"Iter {iteration+1} - Wave {wave_to_fit+1} (Lead {lead_index+1})",
                            x_axis=self._time_points,
                            y_lim=visual_limits
                        )
            
                # 4) RICALCOLO DI M E A per ogni derivazione usando il segnale originale e tutte le onde fittate fino a quel momento
                current_lead_results = self._recalculate_M_and_A(ecg_data, current_lead_results, n_obs)
                
                # 5) AGGIORNAMENTO DELLE ONDE FITTATE CON I PARAMETRI APPENA TROVATI (solo componente oscillatoria, no baseline)
                # di fatto stiamo ancora iterando su ogni lead ed ogni onda, andando a ricostruire l'onda a partire 
                # dai parametri appena stimati (alpha, beta, omega, A) e a popolare fitted_waves. 
                # Ciò non può essere fatto dentro al ciclo del punto 3) perchè _recalculate_M_and_A necessita di tutte le onde
                # fittate in quel momento per poter ricalcolare M e A in modo consistente

                # In fitted_waves andiamo a inserire solo la componente oscillatoria E NON LA BASELINE M, perchè altrimenti 
                # durante la somma all'inzio dell'iterazione successiva sommeremmo la baseline 5 volte (una per ogni onda)
                for lead_index in range(n_leads):
                    for w in range(len(current_lead_results[lead_index]["Alpha"])):
                        alpha = current_lead_results[lead_index]["Alpha"][w]
                        beta = current_lead_results[lead_index]["Beta"][w]
                        omega = current_lead_results[lead_index]["Omega"][w]
                        amp = current_lead_results[lead_index]["A"][w]
                
                        t_star = alpha + 2 * np.arctan2(omega * np.sin((self._time_points - alpha) / 2),
                                                        np.cos((self._time_points - alpha) / 2))
                        
                        fitted_waves[:, lead_index, w] = amp * np.cos(t_star + beta - alpha)

            # 6) I-STEP: WAVE ASSIGNMENT (Labeling)
            current_lead_results = self.assigner.assign_waves(current_lead_results, n_obs, annotation)    

            if self._check_early_stop(current_lead_results):
                lead_results = current_lead_results
                break

            # 7) AGGIORNAMENTO PESI 
            error_weights = self._update_error_weights(ecg_data, fitted_waves, signal_weights)
            
            lead_results = current_lead_results

        
        return lead_results, fitted_waves
