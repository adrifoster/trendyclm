#!/usr/bin/env bash
./xmlchange RUN_TYPE=hybrid
./xmlchange RUN_REFCASE=TRENDY2025_postSASU_20250730-091110
./xmlchange RUN_REFDIR=/glade/derecho/scratch/afoster/TRENDY2025/archive/TRENDY2025_postSASU_20250730-091110/rest/0161-01-01-00000
./xmlchange RUN_REFDATE=0161-01-01
./xmlchange GET_REFCASE=TRUE
./xmlchange DATM_YR_ALIGN=1901
./xmlchange DATM_YR_START=1901
./xmlchange DATM_YR_END=1920
./xmlchange CLM_ACCELERATED_SPINUP="off"
./xmlchange RUN_STARTDATE="1700-01-01"
./xmlchange CLM_BLDNML_OPTS="-bgc bgc -crop -co2_ppmv 277.57"
./xmlchange CLM_FORCE_COLDSTART=off
