import numpy as np
import scipy.optimize
from typing import List, Dict, Tuple
from optimization.utils import plot_ecg_beat
from optimization.fmm_wave import FMMWave
from optimization.fmm_assigner import FMMAssigner

class FMMOptimizer:
    """
    Optimizes Frequency Modulated Moebius (FMM) parameters for multi-lead ECG signals.
    Implements a 3D iterative backfitting algorithm consisting of an M-Step (parameter estimation)
    and an I-Step (wave assignment).
    """
    
    def __init__(self, n_waves: int = 5, max_iter: int = 10, omega_max: float = 0.7):
        """
        Initializes the FMM Optimizer.
        
        Args:
            n_waves: Number of waves to fit (typically 5 for P-Q-R-S-T).
            max_iter: Maximum number of backfitting iterations.
            omega_max: Upper bound for the omega parameter.
        """
        self.n_waves = n_waves
        self.max_iter = max_iter
        self.omega_max = omega_max
        self._grid = None
        self._time_points = None

        # Thresholds from thresholdsMultiFMM_ECG.R
        self.RELEVANT_LEADS = ["I", "II", "V2", "V5"]
        
        # Wave Assigner (I-Step)
        self.assigner = FMMAssigner()

    @staticmethod
    def generate_time_points(n_obs: int) -> np.ndarray:
        """Generates time vectors t in (0, 2*pi] mapped to a single beat."""
        return np.linspace(2 * np.pi / n_obs, 2 * np.pi, n_obs)

    @staticmethod
    def calculate_variance_explained(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculates the Percentage of Variance (PV) explained, equivalent to R2."""
        sse = np.sum((y_true - y_pred)**2)
        sst = np.sum((y_true - np.mean(y_true))**2)
        return 1 - (sse / sst) if sst != 0 else 0.0

    def _initialize_grid(self, n_obs: int):
        """
        Precomputes the Grid Search blocks for alpha and omega pairs.
        This speeds up the global search significantly by pre-calculating pseudo-inverses.
        """
        self._time_points = self.generate_time_points(n_obs)
        
        # Grid dimensions: 48 alphas x 24 omegas = 1152 combinations
        alpha_grid = np.linspace(0, 2 * np.pi, 48)
        omega_grid = np.exp(np.linspace(np.log(0.0001), np.log(1.0), 24))
        
        self._grid = []
        for alpha in alpha_grid:
            for omega in omega_grid:
                # Moebius transformation component
                phi = 2 * np.arctan(omega * np.tan((self._time_points - alpha) / 2))
                cost = np.cos(phi)
                sint = np.sin(phi)
                
                # Design matrix for OLS: intercept, cos(phi), sin(phi)
                m_matrix = np.column_stack((np.ones_like(self._time_points), cost, sint))
                # Pseudo-inverse for rapid OLS solution during grid search
                pinv_m = np.linalg.pinv(m_matrix)
                
                self._grid.append({
                    'alpha': alpha,
                    'omega': omega,
                    'pinv_m': pinv_m,
                    'cost': cost,
                    'sint': sint
                })

    def _evaluate_grid_point(self, grid_item: Dict, residuals_matrix: np.ndarray, weights: np.ndarray) -> Tuple[float, float, float]:
        """Rapidly estimates the RSS for a grid point across all leads simultaneously."""
        # Fast OLS projection: Beta_hat = (M^T M)^-1 M^T * Y
        # residual matrix è il segnale di ground truth (Y_true) da fittare
        pars = grid_item['pinv_m'] @ residuals_matrix
        
        # Prediction Y_hat = M * Beta_hat
        cost = grid_item['cost'][:, np.newaxis]
        sint = grid_item['sint'][:, np.newaxis]
        # Y_hat    = intercetta + omega (m1) * cost + gamma (m2) * sint
        prediction = pars[0, :] + pars[1, :] * cost + pars[2, :] * sint
        
        # Weighted RSS across all leads
        rss_per_lead = np.sum((residuals_matrix - prediction)**2, axis=0)
        total_rss = rss_per_lead @ weights
        
        return grid_item['alpha'], grid_item['omega'], total_rss

    def _global_loss_function(self, params: np.ndarray, residuals_matrix: np.ndarray, weights: np.ndarray) -> float:
        """Loss function (weighted RSS) for fine-tuning alpha and omega using L-BFGS-B."""
        alpha, omega = params
        
        # Moebius transformation
        phi = 2 * np.arctan(omega * np.tan((self._time_points - alpha) / 2))
        m_matrix = np.column_stack((np.ones_like(phi), np.cos(phi), np.sin(phi)))
        
        # Solve for local parameters Beta (Intercept, Delta, Gamma)
        beta_hat, _, _, _ = np.linalg.lstsq(m_matrix, residuals_matrix, rcond=None)
        
        fitted = m_matrix @ beta_hat
        rss_per_lead = np.sum((residuals_matrix - fitted)**2, axis=0)
        return rss_per_lead @ weights

    def _optimize_global_parameters(self, residuals_matrix: np.ndarray, error_weights: np.ndarray) -> Tuple[float, float]:
        """Executes grid search followed by gradient descent refinement to find best Alpha and Omega."""
        best_rss = np.inf
        best_alpha, best_omega = 0.0, 0.0
        
        # 1. Coarse Grid Search
        # per ogni coppia di parametri alpha e omega
        for item in self._grid:
            # risolve il problema di regressione lineare (avendo già a disposizione la pseudo-inversa) e calcola l'errore RSS
            # legato al fitting dell'onda predetta (Y_hat / prediction) rispetto all'onda misurata (Y_true / residuals_matrix)
            alpha, omega, rss = self._evaluate_grid_point(item, residuals_matrix, error_weights)
            if rss < best_rss:
                best_rss = rss
                best_alpha, best_omega = alpha, omega
                
        # 2. Local Refinement using L-BFGS-B
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
        """Estimates lead-specific parameters (M, A, Beta) given global Alpha and Omega."""
        # Calculate phase transformation t*
        t_star = alpha + 2 * np.arctan2(omega * np.sin((self._time_points - alpha) / 2),
                                        np.cos((self._time_points - alpha) / 2))
        
        m_matrix = np.column_stack((np.ones_like(t_star), np.cos(t_star), np.sin(t_star)))
        beta_hat, _, _, _ = np.linalg.lstsq(m_matrix, lead_data, rcond=None)
        
        m_val, delta, gamma = beta_hat
        
        fitted_values = m_val + delta * np.cos(t_star) + gamma * np.sin(t_star)
        pv = self.calculate_variance_explained(lead_data, fitted_values)
        
        # Calculate Amplitude (A) and Phase Shift (Beta)
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
        """Updates error weights based on residual variance for the next iteration."""
        n_obs = ecg_data.shape[0]
        current_fit = np.sum(fitted_waves, axis=2)
        residuals = ecg_data - current_fit
        
        # Calculate residual variance (sigma) per lead
        sigma = np.sum(residuals**2, axis=0) / (n_obs - 1)
        # Weighted inverse of error
        error_weights = (1 / sigma) * signal_weights
        return error_weights / np.sum(error_weights)

    # -----------------------------------------------------------------------
    # Backfitting fitting loop
    # -----------------------------------------------------------------------

    def fit(self, ecg_data: np.ndarray, annotation: float, plot_ecg = False) -> Tuple[List[Dict], np.ndarray]:
        """
        Fits multiple FMM waves to a multi-lead ECG segment using iterative backfitting.
        
        Args:
            ecg_data: Matrix of shape (n_observations, n_leads).
            annotation: Contextual info for wave assignment.
            
        Returns:
            A tuple (parameters_per_lead, fitted_waves_matrix).
        """
        n_obs, n_leads = ecg_data.shape
        self._initialize_grid(n_obs)
        
        # Initialization
        fitted_waves = np.zeros((n_obs, n_leads, self.n_waves))
        signal_weights = np.ones(n_leads) / n_leads
        error_weights = np.ones(n_leads) / n_leads
        
        # Determina i limiti dell'asse Y per la visualizzazione della Lead 1
        lead1_min, lead1_max = np.min(ecg_data[:, 0]), np.max(ecg_data[:, 0])
        y_margin = (lead1_max - lead1_min) * 0.1
        visual_limits = (lead1_min - y_margin, lead1_max + y_margin)

        lead_results = [{"M": [], "A": [], "Alpha": [], "Beta": [], "Omega": [], "Var": []} for _ in range(n_leads)]
        
        for iteration in range(self.max_iter):
            # Backfitting: optimize waves one by one
            # alla prima iterazione cercherà di fittare l'onda con energia maggiore
            # alla seconda fa la stessa cosa ma col segnale residuo
            # e così via per tutte e n_waves onde
            for wave_to_fit in range(self.n_waves):
                # Calculate residuals excluding the current wave being optimized
                mask = np.ones(self.n_waves, dtype=bool)
                mask[wave_to_fit] = False
                # other_waves_sum ha dimensione (n_observations, n_leads), ed è la somma di tutte le onde tranne quella che stiamo fittando
                other_waves_sum = np.sum(fitted_waves[:, :, mask], axis=2)
        
                residuals_matrix = ecg_data - other_waves_sum
            

                # alla prima iterazione, residuals_matrix è uguale a ecg_data perché fitted_waves[:, :, mask] = 0
                # alla seconda iterazione, residuals_matrix sarà ecg_data - prima_onda.best
                # e così via
                # l'ottimizzazione dunque avviene su residual_matrix
                
                # 1. Optimize Global Parameters (Alpha, Omega) shared across leads
                # questi sono validi per tutte le derivazioni 
                opt_alpha, opt_omega = self._optimize_global_parameters(residuals_matrix, error_weights)
                
                # 2. Estimate lead-specific parameters (M, A, Beta\\\\\\\)
                for lead_index in range(n_leads):
                    wave, fitted_vals = self._estimate_local_parameters(residuals_matrix[:, lead_index], opt_alpha, opt_omega)
                    # assegna i valori dell'onda corrente alla matrice fitted_waves
                    fitted_waves[:, lead_index, wave_to_fit] = fitted_vals
                    
                    # Stampiamo il processo di ottimizzazione per la prima derivazione
                    if plot_ecg and lead_index == 0:
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
                    
                    # Store results on the final refinement iteration
                    if iteration == self.max_iter - 1:
                        lead_results[lead_index]["M"].append(wave.M)
                        lead_results[lead_index]["A"].append(wave.A)
                        lead_results[lead_index]["Alpha"].append(wave.alpha)
                        lead_results[lead_index]["Beta"].append(wave.beta)
                        lead_results[lead_index]["Omega"].append(wave.omega)
                        lead_results[lead_index]["Var"].append(wave.variance_explained)
            
            # Re-balance error weights between leads
            error_weights = self._update_error_weights(ecg_data, fitted_waves, signal_weights)

        # I-Step: Wave Assignment
        lead_results = self.assigner.assign_waves(lead_results, n_obs, annotation)
        
        return lead_results, fitted_waves
