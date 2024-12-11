"""Generate slurm script for experiment using settings & schedule."""

import os
import sys
import itertools
import subprocess
from configs import get_experiment_config

# NOTE: could expand this script to parse more slurm args, e.g. username etc.,
# and provide a generic template, but this'll do for now

if __name__ == "__main__":

    # Get experiment settings
    expt_name = str(sys.argv[1])
    expt_type = str(sys.argv[2])

    assert expt_type in ['prior','posterior'], f"Experiment type must be one of 'prior','posterior', {expt_type} given."

    # Get settings and setup parameters for slurm script
    settings, base_params = get_experiment_config(expt_name)
    n_cpus = settings['solver_settings']['threads']

    # get max job index for array job argument (note 0-X inclusive)
    if expt_type == 'prior':
        available_technologies = list(settings['probability_settings']['storage'].keys())
        n_combos = len(list(itertools.combinations(available_technologies, settings['model_settings']['N_technologies'])))
        n_jobs = n_combos
    elif expt_type == 'posterior':
        n_jobs = settings['probability_settings']['n_posterior_samples'] - 1

    # create slurm script for experiment
    slurm_template = 'slurm_submit_cc_template'
    script_name = 'ssub_cc_temp'
    with open(slurm_template,'r') as fread:
        text = fread.read()
    text = text.format(
        expt_name=expt_name,
        expt_type=expt_type,
        n_cpus=n_cpus,
        n_jobs=n_jobs
    )
    with open(script_name,'w') as fout:
        fout.write(text)

    # schedule job using sbatch
    p = subprocess.run(f'sbatch {script_name}',
                       shell=True,
                       check=True,
                       capture_output=True,
                       encoding='utf-8'
                       )
    print(f'Command {p.args} exited with {p.returncode} code\nOutput:\n{p.stdout}')

    # clean up shell script
    os.remove(script_name)

    print(
        "Successfully scheduled jobs with following options:\n"\
        f"Experiment name: {expt_name}\n"\
        f"Experiment type: {expt_type}\n"\
        f"CPUs per task: {n_cpus}\n"\
        f"No. jobs in array: {n_jobs}\n"
    )