# Perform scenario sampling from prior distribution and save to file

import os
import sys
import yaml
import numpy as np

from utils import update_nested_dict
from prob_models import prior_model


if __name__ == "__main__":

    n_prior_scenarios = 64

    expt_id = int(sys.argv[1])

    if expt_id == 0:
        settings_file = 'base_settings'
    elif expt_id == 1:
        settings_file = 'two_tech_settings'
    elif expt_id == 2:
        ... # etc.
    else:
        raise ValueError("Invalid experiment ID.")

    # Load settings for experiment
    with open(os.path.join('configs','base_settings.yaml'), 'r') as f:
        settings = yaml.safe_load(f)
    with open(os.path.join('configs',f'{settings_file}.yaml'), 'r') as f:
        expt_settings = yaml.safe_load(f)
    settings = update_nested_dict(settings, expt_settings)

    # Load base parameter values
    with open(os.path.join('configs','base_params.yaml'), 'r') as f:
        base_params = yaml.safe_load(f)

    # Perform sampling
    np.random.seed(42) # for reproducibility
    theta_scenarios, z_scenarios = prior_model(
        settings['prob_settings'],
        base_params['cost_values'],
        base_params['timeseries_values'],
        base_params['storage_values'],
        n_samples=n_prior_scenarios
    )

    # Save scenarios to file
    save_dir = os.path.join(*settings['results_dir'],'prior','scenarios')
    for i in range(n_prior_scenarios):
        theta_scenarios[i].to_file(os.path.join(save_dir,'thetas',f'scenario_{i}.yaml'))
        z_scenarios[i].to_file(os.path.join(save_dir,'zs',f'scenario_{i}.yaml'))
