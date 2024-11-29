"""Handler functions for setting up experiment configurations."""

import os
import glob
import yaml
from utils import update_nested_dict


def get_experiment_config(expt_name):

    # Load settings for experiment
    try:
        with open(os.path.join('configs',f'{expt_name}_settings.yaml'), 'r') as f:
            expt_settings = yaml.safe_load(f)
    except FileNotFoundError:
        print("Valid experiment names are:")
        for file in glob.glob(os.path.join('configs','*_settings.yaml')):
            print(f"  - {os.path.basename(file).split('_settings')[0]}")
        raise ValueError(f"Invalid experiment name ({expt_name}) used.")

    # Inherit setttings from 'adopted' settings
    if 'adopt' in expt_settings:
        adopted_settings,_ = get_experiment_config(expt_settings.pop('adopt'))
        expt_settings = update_nested_dict(adopted_settings, expt_settings)

    # Load base parameter values
    with open(os.path.join('configs','base_params.yaml'), 'r') as f:
        base_params = yaml.safe_load(f)

    return expt_settings, base_params