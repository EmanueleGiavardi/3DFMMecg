import config
from data_loader import extract_all_vcgs_for_beat
from alignment_strategies import LocalAlignmentStrategy, GlobalLstsqStrategy, GlobalAvgStrategy
import exporter
import metrics_calculator
from sklearn.model_selection import train_test_split
import os

def get_alignment_strategy(strategy_name: str):
    strategies = {
        'local_alignment': LocalAlignmentStrategy,
        'global_alignment_lstsq': GlobalLstsqStrategy,
        'global_alignment_avg': GlobalAvgStrategy
    }
    if strategy_name not in strategies:
        raise ValueError(f"Strategia di allineamento {strategy_name} non supportata.")
    return strategies[strategy_name]()

def split_patients(path_to_R: str):
    patient_codes = [filename[:-12] for filename in sorted(os.listdir(path_to_R))]
    
    train_patients, test_patients = train_test_split(
        patient_codes,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        shuffle=config.SHUFFLE
    )
    return train_patients, test_patients

def main():
    print(f"=== Avvio Pipeline VCG con Strategia: {config.STRATEGY_NAME} ===")

    train_patients, test_patients = split_patients(config.R_PARAMS_DIR)
    
    strategy = get_alignment_strategy(config.STRATEGY_NAME)
    
    def data_extractor_wrapper(patient_code, beat_idx):
        return extract_all_vcgs_for_beat(patient_code, beat_idx, config.ISO_SAMPLES, config.PERCENTILE)

    strategy.fit(
        train_data_extractor=data_extractor_wrapper, 
        train_patients=train_patients, 
        num_beats=config.NUM_BEATS
    )

    # Calcolo VCG per pazienti del test set
    result_vcgs = {}
    
    for patient_code in test_patients:
        print(f"[*] Processing Test Paziente: {patient_code}")
        results_per_patient = []
        
        for beat_idx in range(config.NUM_BEATS):
            vcg_fmm, vcg_gt, vcg_kors = extract_all_vcgs_for_beat(
                patient_code, beat_idx, config.ISO_SAMPLES, config.PERCENTILE
            )
            
            if vcg_fmm is None: 
                continue    

            vcg_fmm_aligned = strategy.align(vcg_fmm, vcg_gt)
            
            results_per_patient.append({
                'beat_idx': beat_idx,
                'VCG_fmm': vcg_fmm_aligned,
                'VCG_kors': vcg_kors,
                'VCG_gt': vcg_gt
            })
            
        result_vcgs[patient_code] = results_per_patient

    # Esportazione dei vcg risultanti in JSON
    print(f"\n[*] Salvataggio risultati")
    exporter.save_to_json(result_vcgs, config.OUTPUT_JSON_PATH)
    
    # Calcolo ed Esportazione Metriche
    df_raw = metrics_calculator.compute_all_metrics(result_vcgs)
    exporter.save_metrics_to_csv(df_raw, config.OUTPUT_RAW_METRICS_CSV)
    
    df_agg = metrics_calculator.create_aggregated_metrics_dataframe(df_raw)
    exporter.save_metrics_to_csv(df_agg, config.OUTPUT_AGG_METRICS_CSV)
    
    print("[*] Pipeline completata")

if __name__ == "__main__":
    main()