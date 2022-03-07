#!/bin/bash
# <Place HPC tool specific headers here>

# SPDX-License-Identifier: MIT

# This bash script is for submitting an array job to a High Performance Computing (HPC) tool such
# as SLURM. Depending on the tool being used, you may only need to change `SLURM_ARRAY_TASK_ID` in
# the last section to the environment variable that is approriate for your HPC tool. Prepend any
# tool specific headers at line 2 above. The total number of array jobs will be
# len(CASE_NAMES) * len(WINDOW_WIDTHS) * len(NOISE_PERCENTS) * len(RUN_COUNTS)

CASE_NAMES=("case_iqr_svd_svht" "case_null" "case_svd_svht")
WINDOW_WIDTHS=({1..7..1})
NOISE_PERCENTS=(0.005 0.01)
RUN_COUNTS=(1 5)

CASE_LEN=${#CASE_NAMES[@]}
WINDOW_LEN=${#WINDOW_WIDTHS[@]}
NOISE_LEN=${#NOISE_PERCENTS[@]}
RUN_LEN=${#RUN_COUNTS[@]}

CPU_COUNT=8

#---------------------------------------------------------------------------------------------------
# Use the following for-loop to see what the script will output for each `TASK_ID`.
# Comment this section out and uncomment the next section when submitting. Alternatively, one could
# change the `echo` output to simply run the suite on a single machine for all parameters.
#---------------------------------------------------------------------------------------------------
TOTAL=$(($CASE_LEN * $WINDOW_LEN * $NOISE_LEN * $RUN_LEN))

for ((TASK_ID = 1; TASK_ID <= $TOTAL; TASK_ID++)); do
    I=$(($TASK_ID - 1))

    CASE_INDEX=$(( $I % $CASE_LEN ))
    WINDOW_INDEX=$(( ($I / $CASE_LEN) % $WINDOW_LEN ))
    NOISE_INDEX=$(( ($I / ($CASE_LEN * $WINDOW_LEN)) % $NOISE_LEN ))
    RUN_INDEX=$(( ($I / ($CASE_LEN * $WINDOW_LEN * $NOISE_LEN)) % $RUN_LEN ))

    CASE_NAME=${CASE_NAMES[$CASE_INDEX]}
    WINDOW_WIDTH=${WINDOW_WIDTHS[$WINDOW_INDEX]}
    NOISE_PERCENT=${NOISE_PERCENTS[$NOISE_INDEX]}
    RUN_COUNT=${RUN_COUNTS[$RUN_INDEX]}
    
    echo "python3 run_case.py ${CASE_NAME} ${CPU_COUNT} ${WINDOW_WIDTH} ${NOISE_PERCENT} ${RUN_COUNT}"
done

#---------------------------------------------------------------------------------------------------
# Use the following section when submiting.
#---------------------------------------------------------------------------------------------------
# Change `SLURM_ARRAY_TASK_ID` to the appropriate environment variable for your HPC tool. If using
# SLURM, then no change is needed.
#I=$(($SLURM_ARRAY_TASK_ID - 1))
#
#CASE_INDEX=$(( $I % $CASE_LEN ))
#WINDOW_INDEX=$(( ($I / $CASE_LEN) % $WINDOW_LEN ))
#NOISE_INDEX=$(( ($I / ($CASE_LEN * $WINDOW_LEN)) % $NOISE_LEN ))
#RUN_INDEX=$(( ($I / ($CASE_LEN * $WINDOW_LEN * $NOISE_LEN)) % $RUN_LEN ))
#
#CASE_NAME=${CASE_NAMES[$CASE_INDEX]}
#WINDOW_WIDTH=${WINDOW_WIDTHS[$WINDOW_INDEX]}
#NOISE_PERCENT=${NOISE_PERCENTS[$NOISE_INDEX]}
#RUN_COUNT=${RUN_COUNTS[$RUN_INDEX]}
#
#python3 run_case.py ${CASE_NAME} ${CPU_COUNT} ${WINDOW_WIDTH} ${NOISE_PERCENT} ${RUN_COUNT}
