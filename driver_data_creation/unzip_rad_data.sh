#!/usr/bin/env bash

TOP_REPO=/glade/campaign/cgd/tss/projects/TRENDY2025
INPUT_DIR=${TOP_REPO}/inputs
RAD_DIR=${INPUT_DIR}/Radiation
RAD_STREAMS_FILE=radiation_streams.txt

export PBS_ACCOUNT=P93300041

# read in radiation stream names
rad_streams=()
while IFS= read -r line; do
  rad_streams+=("$line")
done < ${RAD_STREAMS_FILE}

# change into CRU directory
cd ${RAD_DIR}

# now loop through cru streams
for i in "${rad_streams[@]}"
do
  qcmd -- gunzip ${i}_gcb2025_*.gz &> $i.out &
done
