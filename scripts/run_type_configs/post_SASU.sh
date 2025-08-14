#!/usr/bin/env bash
echo "finidat = 'TRENDY2025_SASUspinup_20250727-134920.clm2.r.0341-01-01-00000.nc '" >> user_nl_clm
./xmlchange RUN_TYPE=hybrid
./xmlchange GET_REFCASE=TRUE
./xmlchange RUN_REFCASE=TRENDY2025_SASUspinup_20250727-134920
./xmlchange RUN_REFDATE=0341-01-01
./xmlchange DRV_RESTART_POINTER=rpointer.cpl.0341-01-01-00000
./xmlchange RUN_REFDIR=/glade/derecho/scratch/afoster/TRENDY2025/archive/TRENDY2025_SASUspinup_20250727-134920/rest/0341-01-01-00000
./xmlchange DATM_YR_ALIGN=1
./xmlchange DATM_YR_START=1901
./xmlchange DATM_YR_END=1920
./xmlchange CLM_ACCELERATED_SPINUP="off"
./xmlchange RUN_STARTDATE="0001-01-01"
./xmlchange CLM_BLDNML_OPTS="-bgc bgc -crop -co2_ppmv 277.57"
./xmlchange MOSART_MODE=NULL
./xmlchange CONTINUE_RUN=FALSE
./xmlchange CLM_FORCE_COLDSTART=off