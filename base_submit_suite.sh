#!/bin/bash
# <Place HPC tool specific headers here>

# SPDX-License-Identifier: MIT

# This bash script is for submitting an array job to a High Performance Computing (HPC) tool such
# as SLURM. Depending on the tool being used, you may only need to change `SLURM_ARRAY_TASK_ID` in
# the last section to the environment variable that is approriate for your HPC tool. Prepend any
# tool specific headers at line 2 above. There are 28 combinations to run.

WINDOW_WIDTHS=({1..7..1})
NOISE_PERCENTS=(0.001 0.002 0.005 0.01)

#---------------------------------------------------------------------------------------------------
# Use the following for-loop to see what the script will output for each `TASK_ID`.
# Comment this section out and uncomment the next section when submitting. Alternatively, one could
# change the `echo` output to simply run the suite on a single machine for all parameters.
#---------------------------------------------------------------------------------------------------
#TOTAL=$((${#WINDOW_WIDTHS[@]} * ${#NOISE_PERCENTS[@]}))
#
#for ((TASK_ID = 1; TASK_ID <= $TOTAL; TASK_ID++)); do
#    I=$((TASK_ID - 1))
#
#    WINDOW_INDEX=$((($I % (${#WINDOW_WIDTHS[@]} * ${#NOISE_PERCENTS[@]})) / ${#NOISE_PERCENTS[@]}))
#    NOISE_INDEX=$(($I % ${#NOISE_PERCENTS[@]}))
#
#    WINDOW_WIDTH=${WINDOW_WIDTHS[$WINDOW_INDEX]}
#    NOISE_PERCENT=${NOISE_PERCENTS[$NOISE_INDEX]}
#    
#    echo "python3 run_suite.py ${WINDOW_WIDTH} ${NOISE_PERCENT}"
#done

#---------------------------------------------------------------------------------------------------
# Use the following section when submiting.
#---------------------------------------------------------------------------------------------------
# Change `SLURM_ARRAY_TASK_ID` to the appropriate environment variable for your HPC tool. If using
# SLURM, then no change is needed.
I=$(($SLURM_ARRAY_TASK_ID - 1))

WINDOW_INDEX=$((($I % (${#WINDOW_WIDTHS[@]} * ${#NOISE_PERCENTS[@]})) / ${#NOISE_PERCENTS[@]}))
NOISE_INDEX=$(($I % ${#NOISE_PERCENTS[@]}))

WINDOW_WIDTH=${WINDOW_WIDTHS[$WINDOW_INDEX]}
NOISE_PERCENT=${NOISE_PERCENTS[$NOISE_INDEX]}

python3 run_suite.py ${WINDOW_WIDTH} ${NOISE_PERCENT}
