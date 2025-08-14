#!/usr/bin/env bash
./xmlchange DATM_YR_ALIGN=1
./xmlchange DATM_YR_START=1901
./xmlchange DATM_YR_END=1920
./xmlchange CLM_ACCELERATED_SPINUP="on"
./xmlchange RUN_STARTDATE="0001-01-01"
./xmlchange CLM_BLDNML_OPTS="-bgc bgc -crop -co2_ppmv 277.57"
./xmlchange MOSART_MODE=NULL
./xmlchange CONTINUE_RUN=FALSE
./xmlchange CLM_FORCE_COLDSTART=on
./xmlchange REST_N=10
./xmlchange DOUT_S_SAVE_INTERIM_RESTART_FILES=TRUE
