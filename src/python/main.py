import argparse
import yaml
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

from data_loader import extract_all_vcgs_for_beat
from alignment_strategies import LocalAlignmentStrategy, GlobalLstsqStrategy, GlobalAvgStrategy
import exporter
import metrics_calculator


def get_alignment_strategy(strategy_name: str):
    strategies = {
        'local_alignment': LocalAlignmentStrategy,
        'global_alignment_lstsq': GlobalLstsqStrategy,
        'global_alignment_avg': GlobalAvgStrategy
    }
    if strategy_name not in strategies:
        raise ValueError(f"Strategia di allineamento '{strategy_name}' non valida.")
    return strategies[strategy_name]()


def split_patients(path_to_R: str, test_size: float, random_state: int, shuffle: bool):
    """Estrae i codici pazienti e li divide in train e test set."""
    patient_codes = [filename[:-12] for filename in sorted(os.listdir(path_to_R))]
    
    train_patients, test_patients = train_test_split(
        patient_codes,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle
    )
    return train_patients, test_patients


def main():
    parser = argparse.ArgumentParser(description="Pipeline di Allineamento VCG")
    parser.add_argument('--config', type=str, required=True, help="Percorso del file config.yaml")
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    params = config['pipeline_params']
    strategy_name = params['strategy_name']
    
    r_params_dir = config['directories']['r_params']
    r_preproc_dir = config['directories']['r_preproc']
    results_dir = Path(config['directories']['results'])
    dataset_dir = Path(config['directories']['dataset'])
    
    strategy_output_dir = results_dir / strategy_name
    strategy_output_dir.mkdir(parents=True, exist_ok=True)
    
    output_json_path = strategy_output_dir / config['output_filenames']['predictions']
    output_raw_metrics = strategy_output_dir / config['output_filenames']['raw_metrics']
    output_agg_metrics = strategy_output_dir / config['output_filenames']['agg_metrics']

    print(f"=== Avvio Pipeline VCG con Strategia: {strategy_name} ===")

    train_patients, test_patients = split_patients(
        path_to_R=r_params_dir,
        test_size=params['test_size'],
        random_state=params['random_state'],
        shuffle=params['shuffle']
    )

    strategy = get_alignment_strategy(strategy_name)
    
    def data_extractor_wrapper(patient_code, beat_idx):
        return extract_all_vcgs_for_beat(
            patient_code, 
            beat_idx, 
            params['iso_samples'], 
            params['percentile'], 
            r_preproc_dir, 
            r_params_dir,
            dataset_dir
        )

    strategy.fit(
        train_data_extractor=data_extractor_wrapper, 
        train_patients=train_patients, 
        num_beats=params['num_beats']
    )

    result_vcgs = {}
    
    for patient_code in test_patients:
        print(f"[*] Processing Test Paziente: {patient_code}")
        results_per_patient = []
        
        for beat_idx in range(params['num_beats']):
            vcg_fmm, vcg_gt, vcg_kors = data_extractor_wrapper(patient_code, beat_idx)
            
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

    print(f"\n[*] Salvataggio risultati JSON in {output_json_path}")
    exporter.save_to_json(result_vcgs, str(output_json_path))
    
    print("[*] Calcolo metriche raw e aggregate...")
    df_raw = metrics_calculator.compute_all_metrics(result_vcgs)
    exporter.save_metrics_to_csv(df_raw, str(output_raw_metrics))
    
    df_agg = metrics_calculator.create_aggregated_metrics_dataframe(df_raw)
    exporter.save_metrics_to_csv(df_agg, str(output_agg_metrics))
    
    print("[*] Pipeline completata con successo!")

if __name__ == "__main__":
    main()