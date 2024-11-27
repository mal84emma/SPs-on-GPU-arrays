# Perform scenario sampling from prior & posterior distributions and save to file

import os
import sys
import time
import numpy as np

from configs import get_experiment_config
from prob_models import prior_model, posterior_model


if __name__ == "__main__":

    expt_id = int(sys.argv[1])
    settings, base_params = get_experiment_config(expt_id)
    prob_settings = settings['probability_settings']

    # ========================================

    start = time.time()

    # Sample from prior
    np.random.seed(42) # for reproducibility
    save_dir = os.path.join(*settings['scenarios_dir'])

    theta_scenarios, z_scenarios = prior_model(
        prob_settings,
        base_params['cost_values'],
        base_params['timeseries_values'],
        base_params['storage_values'],
        n_samples=prob_settings['n_prior_samples']
    )

    for i in range(prob_settings['n_prior_samples']):
        theta_scenarios[i].to_file(os.path.join(save_dir,'thetas',f'scenario_{i}.yaml'))
        z_scenarios[i].to_file(os.path.join(save_dir,'zs',f'scenario_{i}.yaml'))

    print(f"Thetas and zs sampled. ({time.time()-start:.1f}s)")

    # Sample from posterior
    np.random.seed(42) # for reproducibility
    save_dir = os.path.join(save_dir,'varthetas')

    for i,z_scenario in enumerate(z_scenarios):
        vartheta_scenarios = posterior_model(
            z_scenario,
            prob_settings,
            n_samples=prob_settings['n_posterior_samples']
        )

        for j,scenario in enumerate(vartheta_scenarios):
            scenario.to_file(os.path.join(save_dir,f'z_scenario_{i}',f'scenario_{j}.yaml'))

        print(f"z scenario {i} sampled. ({time.time()-start:.1f}s)")