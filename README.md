# **phase-svd**

This repository contains the source code for recreating the research in "Denoising with Singular Value Decomposition for Phase Identification in Power Distribution Systems". The dataset can be downloaded from [Kaggle](https://www.kaggle.com/msk5sdata/phase-svd). Alternatively, the dataset in this work can be recreated from scratch using the [phase-svd-opendss](https://github.com/msk-5s/phase-svd-opendss.git) repository.

## Requirements
    - Python 3.8+ (64-bit)
    - See requirements.txt file for the required python packages.

## Folders
`data/`
: The voltage magnitude dataset, default ckt5 load profiles, synthetic load profiles, and metadata should be placed in this folder (download it from [Kaggle](https://www.kaggle.com/msk5sdata/phase-svd)).

`results/`
: This folder contains the phase identification results for different window widths, noise percents, and run counts. These are the results reported in the paper.

## Running
The `run.py` script can be used to run phase identification for a given snapshot in the year of data. `run_case.py` can be used to run phase identification across the entire year of data for different parameters. If you have access to a computing cluster, then use the `base_submit_case.sh`, which will run `run_case.py` as an array job. `base_submit_case.sh` will need to be modified for the software being used by the computing cluster (see the comments in the script).
> **NOTE: `run_case.py` will save results to the `results/` folder.**
