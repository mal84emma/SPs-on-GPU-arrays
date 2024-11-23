"""Perform initial design of system using prior samples."""

import os
import sys
import shutil

from utils import ScenarioData, get_current_time, get_Gurobi_WLS_env, try_all_designs
from configs import get_experiment_config


if __name__ == "__main__":

    expt_id = int(sys.argv[1])
    settings, base_params = get_experiment_config(expt_id)
    prob_settings = settings['probability_settings']

    # ========================================

    # Load prior samples
    prior_scenarios_dir = os.path.join(*settings['results_dir'],'scenarios','thetas')
    prior_scenarios = [ScenarioData.from_file(os.path.join(prior_scenarios_dir,f'scenario_{i}.yaml')) for i in range(prob_settings['n_prior_samples'])]

    # Set up Gurobi environment
    settings['solver_settings']['env'] = get_Gurobi_WLS_env(silence = not settings['solver_settings']['verbose'])

    # Perform design
    print(f"Starting prior design @ {get_current_time()}")
    save_dir = os.path.join(*settings['results_dir'],'prior')
    best_techs = try_all_designs(prior_scenarios, settings, save_all=save_dir)
    shutil.copy(os.path.join(save_dir,f'{best_techs}_design.yaml'),os.path.join(save_dir,f'best_design.yaml'))
    print(f"Finished prior design @ {get_current_time()}")