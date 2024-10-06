"""Handler functions for setting up experiment configurations."""

import os
import yaml
from utils import update_nested_dict


def get_experiment_config(expt_id):

    # Get experiment name from ID
    if expt_id == 0:
        expt_name = 'base_settings'
    elif expt_id == -1:
        expt_name = 'test_settings'
    elif expt_id == 1:
        expt_name = 'two_tech_settings'
    elif expt_id == 2:
        ... # etc.
    else:
        raise ValueError("Invalid experiment ID.")

    # Load settings for experiment
    with open(os.path.join('configs','base_settings.yaml'), 'r') as f:
        settings = yaml.safe_load(f)
    with open(os.path.join('configs',f'{expt_name}.yaml'), 'r') as f:
        expt_settings = yaml.safe_load(f)
    settings = update_nested_dict(settings, expt_settings)

    # Load base parameter values
    with open(os.path.join('configs','base_params.yaml'), 'r') as f:
        base_params = yaml.safe_load(f)

    return settings, base_params