"""Perform initial design of system using prior samples."""

import os
import sys
import shutil
import itertools

from utils import ScenarioData, get_current_time, get_Gurobi_WLS_env, try_all_designs, solve_model
from configs import get_experiment_config


if __name__ == "__main__":

    expt_id = int(sys.argv[1])
    settings, base_params = get_experiment_config(expt_id)
    prob_settings = settings['probability_settings']

    if len(sys.argv) > 2:
        tech_combo_ind = int(sys.argv[2])
        available_technologies = list(settings['probability_settings']['storage'].keys())
        combos = list(itertools.combinations(available_technologies, settings['model_settings']['N_technologies']))
        tech_combo = combos[tech_combo_ind]
        tech_combo_str = '-'.join(tech_combo)
    else:
        tech_combo = None

    # ========================================

    # Load prior samples
    prior_scenarios_dir = os.path.join(*settings['results_dir'],'scenarios','thetas')
    prior_scenarios = [ScenarioData.from_file(os.path.join(prior_scenarios_dir,f'scenario_{i}.yaml')) for i in range(prob_settings['n_prior_samples'])]

    # Set up Gurobi environment
    settings['solver_settings']['env'] = get_Gurobi_WLS_env(silence = not settings['solver_settings']['verbose'])

    # Perform design
    save_dir = os.path.join(*settings['results_dir'],'prior')

    if tech_combo is None: # do design for all technology combinations together
        print(f"Starting prior design @ {get_current_time()}")
        best_techs = try_all_designs(prior_scenarios, settings, save_all=save_dir)
        shutil.copy(os.path.join(save_dir,f'{best_techs}_design.yaml'),os.path.join(save_dir,f'best_design.yaml'))
        print(f"Finished prior design @ {get_current_time()}")
    else: # just evaluate selected technology combination
        print(f"Starting prior design for tech combo {tech_combo} @ {get_current_time()}")
        settings['model_settings']['storage_technologies'] = list(tech_combo)
        solved_model = solve_model(prior_scenarios, settings)
        solved_model.save_results(os.path.join(save_dir,f'{tech_combo_str}_design.yaml'))
        print(f"Finished prior design for tech combo {tech_combo} @ {get_current_time()}")