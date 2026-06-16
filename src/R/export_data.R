path <- getwd()
source("src/R/requiredFunctionsPreprocessing_v4.1.R")
source("src/R/FMM_ECG3D_Codes/auxMultiFMM_ECG.R", chdir = TRUE)
library(jsonlite)
library(yaml)

project_root <- "."
config <- read_yaml(file.path(project_root, "config/main_config.yaml"))
dataset_dir <- file.path(project_root, config$directories$dataset)
R_preproc_dir <- file.path(project_root, config$directories$r_preproc)
R_params_dir <- file.path(project_root, config$directories$r_params)
csv_files <- list.files(dataset_dir, pattern = "\\.csv$")
patient_codes <- sub("\\.csv$", "", csv_files)

leadName <- c("i","ii","iii","avl","avr","avf","v1","v2","v3","v4","v5","v6")
freqHz <- 500

dir.create(R_preproc_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(R_params_dir, showWarnings = FALSE, recursive = TRUE)

for (patient_code in patient_codes) {
  patient_file <- file.path(dataset_dir, paste0(patient_code, ".csv"))
  if (!file.exists(patient_file)) {
    cat(sprintf("\nSkipping patient: %s (CSV not found)\n", patient_code))
    next
  }
  
  preproc_filename <- file.path(R_preproc_dir, paste0(patient_code, "_PREPROC.json"))
  params_filename <- file.path(R_params_dir, paste0(patient_code, "_PARAMS.json"))
  
  if (file.exists(preproc_filename) && file.exists(params_filename)) {
    cat(sprintf("\nSkipping patient: %s (Outputs already exist)\n", patient_code))
    next
  }
  
  cat("\n========================================\n")
  cat(sprintf("Processing patient: %s\n", patient_code))
  cat("========================================\n")
  
  data <- read.csv(patient_file)
  dataIn <- data[, leadName]
  
  cat("Running global preprocessing...\n")
  preprocessedOutput <- givePreprocessing_git(dataIn = dataIn, freqHz = freqHz)
  pos_results <- preprocessedOutput[[1]]
  m_detrend <- preprocessedOutput[[2]]
  error_preprocessing <- preprocessedOutput[[3]]
  
  all_beats <- unique(pos_results[, "beat_id"])
  
  # --- INIZIO ESPORTAZIONE PREPROCESSING ---
  beats_info <- list()
  for (i in seq_along(all_beats)) {
    selectedBeat <- all_beats[i]
    beatRefsPatient <- pos_results[pos_results[,"beat_id"] == selectedBeat, 
                                   c("iniRef", "finRef", "annoRef"), drop = FALSE]
    if (nrow(beatRefsPatient) > 0) {
      beats_info[[length(beats_info) + 1]] <- list(
        beat_id = unbox(selectedBeat),
        inizio = unbox(as.numeric(beatRefsPatient[1])),
        fine = unbox(as.numeric(beatRefsPatient[2])),
        picco_r = unbox(as.numeric(beatRefsPatient[3]))
      )
    }
  }
  
  preproc_export <- list(
    m_detrend = m_detrend,
    beats = beats_info
  )
  
  write_json(preproc_export, preproc_filename, pretty = TRUE)
  cat(sprintf("Preproc data saved to %s\n", preproc_filename))
  
  # --- CODICE DI OTTIMIZZAZIONE FMM ---
  patient_params_list <- list()
  
  for (i in seq_along(all_beats)) {
    selectedBeat <- all_beats[i]
    beatRefsPatient <- pos_results[pos_results[,"beat_id"] == selectedBeat, 
                                   c("iniRef", "finRef", "annoRef"), drop = FALSE]
    
    if (nrow(beatRefsPatient) == 0) {
      next
    }
    
    selectedBeatData <- m_detrend[as.numeric(beatRefsPatient[1]):as.numeric(beatRefsPatient[2]),]
    annotation_val <- as.numeric(beatRefsPatient[3] - beatRefsPatient[1] + 1)
    
    cat(sprintf("  -> Optimization for beat %d (index %d/%d)...\n", selectedBeat, i, length(all_beats)))
    
    paramsPerLeadPre <- fitMultiFMM_ECG(vDataMatrix = selectedBeatData,
                                        annotation = annotation_val)
    
    if (is.list(paramsPerLeadPre)) {
      paramsPerLeadPre$beat <- unbox(i - 1)
    } else {
      paramsPerLeadPre <- c(list(beat = unbox(i - 1)), as.list(paramsPerLeadPre))
    }
    
    patient_params_list[[length(patient_params_list) + 1]] <- paramsPerLeadPre
  }
  
  write_json(patient_params_list, params_filename, pretty = TRUE)
  cat(sprintf("R results saved to %s\n", params_filename))
  # ----------------------------------------------------------------
}
