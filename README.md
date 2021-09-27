# **phase-svd**

This repository contains the source code for recreating the research in "Denoising with Singular Value Decomposition for Phase Identification in Power Distribution Systems"[^1]. The dataset can be downloaded from [Kaggle](https://www.kaggle.com/msk5sdata/phase-svd). Alternatively, the dataset in this work can be recreated from scratch using the [phase-svd-opendss](https://github.com/msk-5s/phase-svd-opendss.git) repository.

## Requirements
    - Python 3.8+ (64-bit)
    - See requirements.txt file for the required python packages.

## Folders
`data/`
: The voltage magnitude dataset, default ckt5 load profiles, synthetic load profiles, and metadata should be placed in this folder (download it from [Kaggle](https://www.kaggle.com/msk5sdata/phase-svd)).

`results/`
: This folder contains the phase identification results for different window widths and noise percents. These are the results reported in the paper.

`stats/`
: This folder contains summary statistics of the results. These are the summary statistics reported in the paper.

## Running
The `run.py` script can be used to run phase identification for a given snapshot in the year of data. `run_suite.py` can be used to run phase identification across the entire year of data for different window widths and noise percents. If you have access to a computing cluster, then use the `base_submit_suite.sh`, which will run `run_suite.py` as an array job. `base_submit_suite.sh` will need to be modified for the software being used by the computing cluster (see the comments in the script).
> **NOTE: `run_suite.py` will save results to the `results/` folder.**

## Citation (BibTex)
If you found this repository useful in your own research, then feel free to cite the following paper:
```
@misc{zaragoza_rao_2021,
    title={Denoising with Singular Value Decomposition for Phase Identification in Power Distribution Systems},
    url={https://www.techrxiv.org/articles/preprint/Denoising_with_Singular_Value_Decomposition_for_Phase_Identification_in_Power_Distribution_Systems/15102072/2},
    DOI={10.36227/techrxiv.15102072.v2},
    publisher={TechRxiv},
    author={Zaragoza, Nicholas and Rao, Vittal},
    year={2021},
    month={Aug}
} 
```

[^1]: Zaragoza, Nicholas; Rao, Vittal (2021): Denoising with Singular Value Decomposition for Phase Identification in Power Distribution Systems. TechRxiv. Preprint. https://doi.org/10.36227/techrxiv.15102072.v2
