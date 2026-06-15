"""
File di configurazione centrale per la pipeline VCG.
"""

# STRATEGIA
# Opzioni disponibili: 'local_alignment', 'global_alignment_lstsq', 'global_alignment_avg'
STRATEGY_NAME = 'global_alignment_avg'

# DATASET e PAZIENTI

R_PARAMS_DIR = '../../R_params' # directory con i parametri FMM per ogni paziente
R_PREPROC_DIR = '../../R_preproc' # directory con i parametri di preprocessing per ogni paziente
TEST_SIZE = 0.3
RANDOM_STATE = 42
SHUFFLE = True

# PARAMETRI FMM e VCG PRE-PROCESSING
NUM_BEATS = 30          # Numero di battiti da estrarre per ogni paziente
ISO_SAMPLES = 20        # Campioni per il centraggio del VCG
PERCENTILE = 0.95       # Percentile per la normalizzazione

# --- PERCORSI DI OUTPUT ---
RESULTS_DIR = '../../results' # directory per il salvataggio dei risultati


OUTPUT_JSON_PATH = f'{RESULTS_DIR}/{STRATEGY_NAME}/vcg_predictions.json'
OUTPUT_RAW_METRICS_CSV = f'{RESULTS_DIR}/{STRATEGY_NAME}/metrics_raw.csv'
OUTPUT_AGG_METRICS_CSV = f'{RESULTS_DIR}/{STRATEGY_NAME}/metrics_aggregated.csv'