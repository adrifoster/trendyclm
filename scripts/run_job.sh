#!/usr/bin/env bash

#==============================================================================
#
# Script Name: run_job
# Description: Creates and configures a new CTSM case.
# Usage: ./run_job <config_file>
#
# Arguments:
#   <config_file>     Path to the configuration file containing environment variables.
#
# Requirements:
#   - Must have the necessary CTSM environment set up.
#   - Requires a valid configuration file.
#   - The configuration file must define necessary environment variables.
#
# Environment Variables (loaded from config file):
#   CASE_DIR          Directory where cases will be created.
#   CASE_NAME         Name of case.
#   SRC_DIR           Directory containing cime scripts.
#   STOP_N            Number of years for each submission of the simulation.
#   RESUBMIT          Number of resubmissions.
#   COMPSET           Compset to use
#   RES               Model resolution to use
#   USER_NL_CLM_FILE  user_nl_clm file name to use
#   USER_NL_DATM_FILE user_nl_datm_streams file name to use
#   PROJECT           Project charge number.
#   USER_NL_DIR       Directory containing user_nl_* files
#   OUT_DIR           Directory where output files will be placed
#   RUN_TYPE          run type (e.g., AD_spinup)

#
# Example:
#   ./setup_run config_file.cfg
#
# Author: Adrianna Foster
# Date: 2025-03-14
# Version: 1.0
#
#==============================================================================\

if [ $# -lt 1 ]
then
  echo "ERROR: please specify config file"
  exit 1
fi
config_file="$1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# load utility functions
source "${SCRIPT_DIR}/common_utils"

# read and export config variables
read_config "$config_file"

# ensure required environment variables are set
required_vars=(
  CASE_DIR CASE_NAME SRC_DIR STOP_N 
  RESUBMIT COMPSET RES USER_NL_CLM_FILE 
  USER_NL_DATM_FILE PROJECT USER_NL_DIR RUN_TYPE OUT_DIR
  )
check_required_vars "${required_vars[@]}"

# create timestamped case name
timestamp="$(date '+%Y%m%d-%H%M%S')"
case_root=${CASE_DIR}/${CASE_NAME}_${timestamp}

if [[ -d "$case_root" ]]; then
  echo "ERROR: Case directory ${case_root} already exists."
  exit 1
fi

# change into cime directory
cd "${SRC_DIR}/cime/scripts" || { echo "ERROR: Failed to change directory to ${SRC_DIR}/cime/scripts"; exit 1; }

# create case
./create_newcase --case ${case_root} --compset ${COMPSET}  --res ${RES} --project ${PROJECT} --run-unsupported --output-root ${OUT_DIR}
cd "${case_root}" || { echo "ERROR: Failed to change directory to ${case_name}"; exit 1; }

# copy user_nl_files
copy_user_nl_file "${USER_NL_DIR}/${USER_NL_CLM_FILE}" "user_nl_clm"
copy_user_nl_file "${USER_NL_DIR}/user_nl_cpl" "user_nl_cpl"
copy_user_nl_file "${USER_NL_DIR}/user_nl_mosart" "user_nl_mosart"
copy_user_nl_file "${USER_NL_DIR}/${USER_NL_DATM_FILE}" "user_nl_datm_streams"

# XML settings (common)
./xmlchange STOP_OPTION="nyears"
./xmlchange DOUT_S=TRUE
./xmlchange --subgroup case.run JOB_WALLCLOCK_TIME=12:00:00
./xmlchange --subgroup case.st_archive JOB_WALLCLOCK_TIME=01:00:00
./xmlchange STOP_N=${STOP_N}
./xmlchange RESUBMIT=${RESUBMIT}

# apply RUN_TYPE-specific logic
RT_SCRIPT="${SCRIPT_DIR}/run_type_configs/${RUN_TYPE}.sh"
if [[ -f "$RT_SCRIPT" ]]; then
  source "$RT_SCRIPT"
else
  echo "ERROR: Unknown RUN_TYPE or missing config script: $RUN_TYPE"
  exit 1
fi

# finalize and submit
./case.setup
./preview_namelists
./case.build
./case.submit

