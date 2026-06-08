import json
import numpy as np
import pandas as pd
import os

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def save_to_json(data_dict: dict, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(data_dict, f, cls=NumpyEncoder, indent=2)
    print(f"[+] Dati VCG salvati con successo in: {filepath}")

def save_metrics_to_csv(df_metrics: pd.DataFrame, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df_metrics.to_csv(filepath, index=False)
    print(f"[+] Metriche esportate con successo in: {filepath}")