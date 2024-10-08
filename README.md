Install `linopy` and `cmdstanpy` using conda to ensure proper compilation of libraries.
Note, this needs to be done **before** pip installing the rest of the packages to ensure compatible libraries are installed.

Install steps,
```
conda create env -n EP-VOI python=3.9
conda activate EP-VOI
conda install -c conda-forge linopy=0.3.14
conda install -c conda-forge cmdstanpy=1.2.4
pip install -r requirements.txt
```

For machine where only SP solving to to be performed, install `linopy=0.3.14` and use `reqs-solve-only.txt` for minimal install.