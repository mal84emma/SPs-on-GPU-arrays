Install `linopy` and `cmdstanpy` using conda to ensure proper compilation of libraries.
Note, this needs to be done **before** pip installing the rest of the packages to ensure compatible libraries are installed.

Install steps,
```
conda create env -n EP-VOI python=3.9
conda activate EP-VOI
conda install -c conda-forge linopy
conda install -c conda-forge cmdstanpy
pip install -r requirements.txt
```