#!/usr/bin/env bash

TOP_REPO=/glade/campaign/cgd/tss/projects/TRENDY2025 
INPUT_DIR=${TOP_REPO}/inputs
CRU_DIR=${INPUT_DIR}/crujra3
CRU_STREAMS_FILE=cru_streams.txt
CRU_STREAMS_PREFIX=crujra.v3.5d.

export PBS_ACCOUNT=P93300041

# read in cru stream names
cru_streams=()
while IFS= read -r line; do
  cru_streams+=("$line")
done < ${CRU_STREAMS_FILE}

# change into CRU directory
cd ${CRU_DIR}

# now loop through cru streams
for i in "${cru_streams[@]}"
do
  qcmd -- gunzip ${CRU_STREAMS_PREFIX}$i.*.gz &> $i.out &
done
