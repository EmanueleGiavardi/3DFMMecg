# Mapping Cardiac Anatomy Through The Lens Of Vision

Questo repository contiene la pipeline computazionale per l'estrazione, la ricostruzione e l'allineamento spaziale del Vettorcardiogramma (VCG) a partire da segnali ECG, utilizzando il modello parametrico 3D-FMM. 
Il progetto valuta diverse strategie di allineamento spaziale (Globali vs. Locali) confrontandole con le baseline cliniche (metodo di Kors), calcolando metriche morfologiche e vettoriali per valutarne l'accuratezza rispetto al Ground Truth (sistema di Frank).

---

## 📂 Struttura del Progetto

```text
3DFMMecg/
├── config/
│   └── main_config.yaml             # File di configurazione centrale (R e Python)
├── data/
│   ├── PTB/                         # Segnali raw del database PTB
│   ├── R_params/                    # Output R: Parametri estratti dal modello FMM
│   └── R_preproc/                   # Output R: Dati di pre-processing e detrending
├── notebooks/
│   ├── pipeline_single_patient.ipynb # Demo interattiva della pipeline su singolo paziente
│   └── visualization.ipynb          # Notebook per la generazione di grafici e box plot
├── results/                         # Directory principale di output
│   ├── global_alignment_avg/        # Risultati metrica globale (Media)
│   ├── global_alignment_lstsq/      # Risultati metrica globale (Minimi Quadrati)
│   ├── local_alignment/             # Risultati metrica locale (Paziente-Specifico)
│   ├── plots/                       # Immagini e grafici esportati
│   └── test_allineamento_gt_train/  # Analisi PCA sul disallineamento del Ground Truth
├── src/
│   ├── python/                      # Moduli e script Python
│   │   ├── vcg/                     # Package per calcolo e visualizzazione VCG
│   │   ├── alignment_strategies.py  
│   │   ├── data_loader.py           
│   │   ├── exporter.py              
│   │   ├── gt_alignment_analysis.py # Script per analisi del Ground Truth
│   │   ├── main.py                  # Script principale di esecuzione pipeline
│   │   ├── metrics_calculator.py    
│   │   └── utils.py                 
│   └── R/                           # Script e dipendenze R
│       ├── FMM_ECG3D_Codes/         # Moduli originali FMM
│       ├── export_data.R            # Script principale per estrazione parametri
│       ├── fittingExample.R         
│       ├── install_requirements.R   # Script di installazione librerie R
│       └── requiredFunctionsPreprocessing_v4.1.R
├── README.md
└── requirements.txt                 # Dipendenze Python
```

## 📥 Prerequisiti e Installazione

L'esecuzione della pipeline richiede la configurazione di due ambienti separati (R e Python).
1. Ambiente R
- È necessario avere installato R in versione 4.6.0.
- Eseguire lo script dedicato per installare tutte le librerie necessarie:
```
Rscript src/R/install_requirements.R
```
2. Ambiente Python
- Creare e attivare un virtual environment utilizzando Python 3.14:
```
python3.14 -m venv 3DFMMecg_venv
source 3DFMMecg_venv/bin/activate
```
- Installare le dipendenze richieste:
```
pip install -r requirements.txt
```

## ⚙️ Esecuzione della Pipeline

**Attenzione**: Tutti i comandi seguenti devono essere lanciati dalla root directory del progetto, (`3DFMMecg/`).
I parametri di esecuzione, le directory di input/output e le impostazioni degli esperimenti sono centralizzati nel file ```config/main_config.yaml```. Modificare questo file per cambiare strategia di allineamento o altri parametri della pipeline.

### Fase 1: Estrazione Parametri (R)

Il primo step elabora i segnali ECG originali, applica il pre-processing ed esegue il fitting del modello 3D-FMM per estrarre i parametri descrittivi delle onde (posto di aver salvato i file ```sXXXX_re.csv``` in ```data/PTB```).

```
Rscript src/R/export_data.R
```
⚠️ **NOTA COMPUTAZIONALE**: *questo step ha impiegato circa 18 ore su CPU: 11th Gen Intel i7-11700K (16) @ 4.900GHz, estraendo dati di preprocessing e parametri FMM di TUTTI i battiti (in media circa 150) per i primi 31 pazienti*.

I file generati verranno salvati nelle cartelle ```data/R_preproc/``` e ```data/R_params/``` (come definito nel config).

### Fase 2: Creazione, Allineamento VCG e Calcolo Metriche (Python)

Questo script carica i parametri calcolati da R, ricostruisce il VCG spaziale, applica la strategia di allineamento (globale o locale, anch'essa definita nel config) e calcola le metriche di errore vettoriali e morfologiche.

```
python src/python/main.py --config config/main_config.yaml
```

I risultati vengono salvati dinamicamente all'interno di ```results/{nome_strategia}/``` (ad es. ```results/local_alignment/```). Per ogni run verranno generati:

* ```vcg_predictions.json```: Struttura contenente le coordinate spaziali 3D dei VCG elaborati.
* ```metrics_raw.csv```: Metriche dettagliate calcolate per ogni singolo battito di ogni singolo paziente.
* ```metrics_aggregated.csv```: Statistiche aggregate (Media, Deviazione Standard) per paziente.

### Fase 3: Visualizzazione e Analisi (Jupyter)

L'analisi visiva e la generazione dei report grafici avvengono in ambiente Notebook.

Lanciare Jupyter Notebook:
```
jupyter notebook
```

* ```notebooks/visualization.ipynb```: Legge i file CSV generati in Fase 2 e produce i box plot comparativi e le analisi quantitative. Tutti i grafici vengono salvati automaticamente e organizzati per categoria (picco, morfologia) all'interno di ```results/plots/```.

* ```notebooks/pipeline_single_patient.ipynb```: Un notebook interattivo progettato per esplorare la pipeline completa e visualizzare i loop spaziali 3D su un singolo paziente a scopo dimostrativo.


### Script di Analisi Disallineamento del Ground Truth (PCA)
Questo script aggiuntivo esegue un'analisi delle Componenti Principali (PCA) sui VCG reali di Frank appartenenti ai pazienti del Train Set. Lo scopo è quantificare e visualizzare la varianza spaziale fisiologica (il disallineamento intrinseco) dell'asse elettrico tra soggetti diversi. Dal punto di vista metodologico, questo passaggio serve a dimostrare empiricamente perché l'utilizzo di una singola matrice di trasformazione globale sia inefficace.

Per lanciare l'analisi, eseguire dalla root:
```bash
python src/python/gt_alignment_analysis.py --config config/main_config.yaml
```

I risultati visivi verranno estratti e salvati nella directory results/test_allineamento_gt_train/. Al suo interno:
- ```allineamento_pca_gt.png```: Un grafico 3D riassuntivo che mostra la sovrapposizione e la dispersione dei vettori direzionali principali (PC1) di tutti i pazienti.
- ```sXXXX_re.png```: Grafici 3D individuali per ciascun paziente, raffiguranti lo sviluppo dei rispettivi loop VCG di Ground Truth nello spazio.
