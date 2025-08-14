#!/usr/bin/env bash

#==============================================================================
# File: common_utils
# Description: A collection of utility functions for CTSM ensemble scripts.
#              This script provides functions for environment validation,
#              configuration parsing, and other common operations.
#
# Usage:
#   source /path/to/common_utils
#
# Functions:
#   check_required_vars VAR1 VAR2 ...   - Ensures environment variables are set.
#   read_config <config_file>           - Parses key-value pairs from a config file.
#
# Notes:
# - This script is intended to be *sourced* from other scripts, not executed directly.
# - Use `check_required_vars` to validate required environment variables.
# - `read_config` should be used to parse configuration files safely.
#
# Example:
#   source common_utils
#   check_required_vars CASE_DIR SRC_DIR
#   read_config my_config.cfg
#
# Author: Adrianna Foster
# Date: 2025-03-14
# Version: 1.0
#
#==============================================================================

# function to check required variables
check_required_vars() {
  local missing=false
  for var in "$@"; do
    if [[ -z "${!var}" ]]; then
      echo "ERROR: $var is not set"
      missing=true
    fi
  done
  $missing && exit 1
}

# function to read a config file and export variables
read_config() {
  local config_file="$1"

  if [[ ! -f "$config_file" ]]; then
    echo "ERROR: Config file '$config_file' not found!"
    exit 1
  fi

  # Read key-value pairs, ignoring comments and blank lines
  while IFS='=' read -r key value; do
  
    # ignore empty lines and comments
    if [[ -z "$key" || "$key" =~ ^# ]]; then
      continue
    fi
    
    # trim spaces
    key=$(echo "$key" | tr -d '[:space:]')
    value=$(echo "$value" | sed 's/^ *//;s/ *$//')

    # remove surrounding quotes (single or double)
    value=$(echo "$value" | sed 's/^["'\''"]//;s/["'\''"]$//')
    
    # assign to a variable using `declare`
    export "$key=$value"
    
  done < "$config_file"
}

copy_user_nl_file() {
  local src_file="$1"
  local dest_file="$2"
  if [[ ! -f "${src_file}" ]]; then
    echo "ERROR: ${src_file} not found!"
    exit 1
  fi
  cp "${src_file}" "${dest_file}"
}