# Perform posterior design of system for each scenario samples from prior

import os
import sys
import yaml
import time
import warnings

from tqdm import tqdm
import multiprocess as mp
from functools import partial

from energy_model import EnergyModel
from prob_models import posterior_model
from utils import ScenarioData, update_nested_dict, get_Gurobi_WLS_env, get_current_time



def posterior_design(measured_scenario_tuple, settings):
    """TODO: function to perform posterior sampling and system design
    both with and without storage optionality. Perform both designs
    in single function (process) to all sample sharing."""

    i,measured_scenario = measured_scenario_tuple # unpack prior scenario

    # Set up Gurobi environment
    env = get_Gurobi_WLS_env(silence=True)
    if env is not None: solver_kwargs = {'solver_name':'gurobi','env':env}
    else: solver_kwargs = {'solver_name': 'highs'}

    # Sample from posterior
    # =====================
    # note: posterior samples used by both design options
    print(f"Generating posterior samples for scenario {i} @ {get_current_time()}")
    posterior_scenarios = posterior_model(measured_scenario, settings['prob_settings']) # n_samples=settings['n_posterior_samples']

    for m,scenario in enumerate(posterior_scenarios): # save scenarios to file
        scenario.to_file(os.path.join(*settings['results_dir'],'posterior','scenarios',f'scenario_{m}.yaml'))

    # Perform design with optionality (free)
    # ======================================
    print(f"Starting open design for scenario {i} @ {get_current_time()}")
    model = EnergyModel()
    with warnings.catch_warnings():
        # filter pandas warnings, `DeprecationWarning: np.find_common_type is deprecated.`
        warnings.simplefilter("ignore", category=FutureWarning)
        model.generate_SP(posterior_scenarios,settings['model_settings'])
        model.solve(**solver_kwargs)
    model.save_results(os.path.join(*settings['results_dir'],'posterior','open_design.yaml'))

    # Perform design without optionality (restricted)
    # ===============================================
    print(f"Starting restricted design for scenario {i} @ {get_current_time()}")
    # Load storage technologies selected in prior design
    with open(os.path.join(*settings['results_dir'],'prior','design.yaml'), 'r') as f:
        prior_design = yaml.safe_load(f)
    selected_technologies = [k for k,v in prior_design['design']['selected_technologies'].items() if v]
    settings['model_settings']['technologies_to_use'] = selected_technologies # update model settings

    model = EnergyModel()
    with warnings.catch_warnings():
        # filter pandas warnings, `DeprecationWarning: np.find_common_type is deprecated.`
        warnings.simplefilter("ignore", category=FutureWarning)
        model.generate_SP(posterior_scenarios,settings['model_settings'])
        model.solve(**solver_kwargs)
    model.save_results(os.path.join(*settings['results_dir'],'posterior','restricted_design.yaml'))

    print(f"Finished posterior designs for scenario {i} @ {get_current_time()}")


if __name__ == "__main__":

    # Run params
    n_measured_scenarios = 64
    n_concurrent_designs = 4
    offset = 0


    expt_id = int(sys.argv[1])

    if len(sys.argv) > 2: scenario_id = int(sys.argv[2])
    else: scenario_id = None

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
    measured_scenarios_dir = os.path.join(*settings['results_dir'],'prior','scenarios','zs')
    measured_scenarios = [ScenarioData.from_file(os.path.join(measured_scenarios_dir,f'scenario_{i}.yaml')) for i in range(n_measured_scenarios)]

    design_wrapper = partial(posterior_design,settings=settings)
    scenarios_to_design = list(enumerate(measured_scenarios))[offset:n_measured_scenarios+offset]

    if scenario_id is not None: # scenario from command line (for script batching)
        posterior_design((scenario_id,measured_scenarios[scenario_id]), settings)
    else:
        if n_concurrent_designs > 1: # parallel processing
            with mp.Pool(n_concurrent_designs) as pool:
                design_results = list(tqdm(pool.imap(design_wrapper, scenarios_to_design), total=len(scenarios_to_design)))
        else: # serial processing
            design_results = [design_wrapper(t) for t in tqdm([(None,s) for s in scenarios_to_design])]
