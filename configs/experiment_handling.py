"""Handler functions for setting up experiment configurations."""

import os
import yaml
from utils import update_nested_dict


def get_experiment_config(expt_id):

    # Get experiment name from ID
    if expt_id == 1:
        expt_name = 'base_settings'
    elif expt_id == 11:
        expt_name = 'epsilon_p20_settings'
    elif expt_id == 12:
        expt_name = 'epsilon_p30_settings'
    elif expt_id == 13:
        expt_name = 'epsilon_p10_settings'
    elif expt_id == 2:
        expt_name = 'two_techs_settings'
    elif expt_id == 3:
        expt_name = 'CVaR_settings'
    elif expt_id == 31:
        expt_name = 'CVaR_a10_b2_settings'
    elif expt_id == 32:
        expt_name = 'CVaR_a10_b5_settings'
    elif expt_id == 33:
        expt_name = 'CVaR_a25_b1_settings'
    elif expt_id == 34:
        expt_name = 'CVaR_a25_b2_settings'
    elif expt_id == -1:
        expt_name = 'test_settings'
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