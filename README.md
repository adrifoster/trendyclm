# TRENDY-CLM

Scripts, configs, and utilities for running TRENDY simulations with CLM.

## Structure
- `configs/`: CLM case configs
- `driver_data_creation/`: Driver data prep scripts
- `user_mods/`: User namelist files
- `scripts/`: Job creation and helper scripts
- `notebooks/`: Post-processing & TRENDY criteria verification notebooks
- `data/`: Metadata like required output variable lists

## Usage
1. Prepare driver data (`driver_data_creation/`).
2. Create and submit jobs (`scripts/run_job.sh`).
3. Post-process outputs (`notebooks/`).