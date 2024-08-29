# Perform initial design of system using prior samples

# Perform posterior design of system for each scenario samples from prior

import os
import sys
import yaml
import time
import warnings

from energy_model import EnergyModel
from utils import ScenarioData, update_nested_dict, get_Gurobi_WLS_env, get_current_time


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

    # ========================================

    # Load prior samples
    prior_scenarios_dir = os.path.join(*settings['results_dir'],'prior','scenarios','thetas')
    prior_scenarios = [ScenarioData.from_file(os.path.join(prior_scenarios_dir,f'scenario_{i}.yaml')) for i in range(n_prior_scenarios)]

    # Perform design
    # ==============
    print(f"Starting prior design @ {get_current_time()}")

    # Set up Gurobi environment
    env = get_Gurobi_WLS_env(silence=True)
    if env is not None: solver_kwargs = {'solver_name':'gurobi','env':env}
    else: solver_kwargs = {'solver_name': 'highs'}

    # Setup up model and solve
    model = EnergyModel()
    with warnings.catch_warnings():
        # filter pandas warnings, `DeprecationWarning: np.find_common_type is deprecated.`
        warnings.simplefilter("ignore", category=FutureWarning)
        model.generate_SP(prior_scenarios,settings['model_settings'])
        model.solve(**solver_kwargs)
    model.save_results(os.path.join(*settings['results_dir'],'prior','design.yaml'))

    print(f"Finished prior design @ {get_current_time()}")