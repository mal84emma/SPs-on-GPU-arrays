# The value of hedging against energy storage uncertainties when designing energy parks

This repository supports the article 'The value of hedging against energy storage uncertainties when designing energy parks', which is available online at https://arxiv.org/abs/2503.15416. It provides the code and data used to perform the numerical experiments in the paper.

## Requirements

A suitable environment for running this code can be initialised using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#) as follows:

```
conda create --name EPvoi python=3.9
conda activate EPvoi
conda install -c conda-forge linopy=0.3.14
conda install -c conda-forge cmdstanpy=1.2.4
pip install -r requirements.txt
```

Note:
- `linopy` and `cmdstanpy` need to be installed from conda-forge to get the link C++ backends to install (compile) properly.
- As there large numbers of scenario and results files are generated these are provided separately at https://zenodo.org/records/15050619.

## Solver license files

The experiments in the paper used the Gurobi solver for speed, using a WLS license. However, the code also supports the use of the open-source solver [HiGHS](https://highs.dev/).

You can get a free academic WLS license for Gurobi [here](https://www.gurobi.com/academia/academic-program-and-licenses/). Update the template license file `resources/gurobi.lic` to enable the Gurobi solver. To prevent your `gurobi.lic` file from being tracked by the repository, use the command 'git update-index --assume-unchanged FILE_NAME'.

## Running experiments

All experiment scripts should be run from root dir using the syntax,

```
python -m experiments.{fname} {expt_name}
```

there should be a corresponding experiment settings file in the `configs` directory named `{expt_name}_settings.yaml`, which specifies the settings to be used for the experiment.

There are 3 experiment scripts that should be run in the following order:
1. `sample_scenarios` - generates scenarios for the experiment with specified statistical parameters
2. `prior_design` - performs energy system design with each storage technology using scenarios sampled from prior distributions
3. `posterior_design` - performs energy system design with each storage technology using scenarios sampled from posterior distribution corresponding to each prior sample