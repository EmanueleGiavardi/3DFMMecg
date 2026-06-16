# R dependencies for FMM-Applications-3DECG

packages_to_install <- c(
  "jsonlite", 
  "stringr", 
  "rsleep", 
  "RColorBrewer", 
  "FMM",
  "yaml"
)

# Identify missing packages
new_packages <- packages_to_install[!(packages_to_install %in% installed.packages()[,"Package"])]

if(length(new_packages)) {
  cat("Installing missing packages: ", paste(new_packages, collapse = ", "), "\n")
  install.packages(new_packages, repos = "http://cran.us.r-project.org")
} else {
  cat("All required packages are already installed.\n")
}
