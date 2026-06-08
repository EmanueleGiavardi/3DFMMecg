"""
File di configurazione centrale per la pipeline VCG.
"""

# --- IMPOSTAZIONI STRATEGIA ---
# Opzioni disponibili: 'local_alignment', 'global_alignment_lstsq', 'global_alignment_avg'
STRATEGY_NAME = 'global_alignment_avg'

# --- DATASET & PAZIENTI ---
# TODO: scrivere modulo train_test_split
R_PARAMS_DIR = '../../R_params'
TEST_SIZE = 0.3
RANDOM_STATE = 42
SHUFFLE = True

# --- PARAMETRI DI PRE-PROCESSING FMM & VCG ---
NUM_BEATS = 30  # Numero di battiti da estrarre per ogni paziente
ISO_SAMPLES = 20        # Campioni isoelettrici per il centraggio
PERCENTILE = 0.95       # Percentile per la normalizzazione

# --- PERCORSI DI OUTPUT ---
OUTPUT_JSON_PATH = f'../../results/{STRATEGY_NAME}/vcg_predictions.json'
OUTPUT_RAW_METRICS_CSV = f'../../results/{STRATEGY_NAME}/metrics_raw.csv'
OUTPUT_AGG_METRICS_CSV = f'../../results/{STRATEGY_NAME}/metrics_aggregated.csv'