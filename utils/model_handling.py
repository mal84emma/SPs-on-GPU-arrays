"""Helper functions for working with models."""

import os
import time
import warnings
import itertools

from typing import List, Dict
import numpy as np
from utils import ScenarioData
from scenarioReducer import Fast_forward
from energy_model import EnergyModel


def reduce_scenarios(
        scenarios: List[ScenarioData],
        settings: Dict
    ) -> List[ScenarioData]:
    """Reduce scenarios using Fast Forward algorithm, with indivudal
    optimized costs as statistical metric, and :math:`L_2` distance.

    Args:
        scenarios (List[ScenarioData]): List of Scenario objects defining
            time series data to use in Scenario Programming model.
        settings (Dict): Dictionary of settings used to construct model.
            See configs/base_settings.yaml for required keys.

    Returns:
        List[ScenarioData]: List of reduced scenarios (ScenarioData objects).
    """

    M = len(scenarios)
    Nred = settings['probability_settings']['n_reduced_scenarios']

    print(f"Reducing {M} scenarios to {Nred} using Fast Forward algorithm.")
    start = time.time()

    # get scenario probabilities
    if all([scenario.probability is not None for scenario in scenarios]):
        probs = np.array([scenario.probability for scenario in scenarios])
        assert np.isclose(np.sum(probs), 1.0, rtol=1e-3),\
            f"Scenario weightings must sum to 1. Currently sum to {np.sum(probs)}"
    else: # assume scenarios equally probable
        probs = np.ones(M)/M
    # reset scenario object probabilities for individual evals
    for scenario in scenarios:
        scenario.probability = None

    # evaluate optimized cost for each scenario
    if (env := settings['solver_settings'].get('env', None)): # temporarily suppress Gurobi output
        env.setParam('OutputFlag',0)
    else:
        settings['solver_settings']['log_to_console'] = False

    indiv_op_costs = []
    for m,scenario in enumerate(scenarios):
        model = solve_model([scenario], settings)
        indiv_op_costs.append(model.corrected_objective)
        print(f"Scenario {m} obj. ({time.time()-start:.1f}s): {model.corrected_objective}")
    indiv_op_costs = np.array([indiv_op_costs]) # needs to be 2d np.array

    # perform reduction using fast forward algorithm
    FFreducer = Fast_forward(indiv_op_costs, probs)
    reduced_scenario_stats, reduced_probs, reduced_indices = FFreducer.reduce(distance=2,n_scenarios=Nred)
    # sort indices and probs by indices
    reduced_probs = [prob for _,prob in sorted(zip(reduced_indices,reduced_probs))]
    reduced_indices = sorted(reduced_indices)
    print(reduced_scenario_stats)
    print(reduced_indices)
    print(reduced_probs)

    # get reduced scenarios and assign probabilities and ids
    reduced_scenarios = [scenarios[ind] for ind in reduced_indices]
    for i in range(Nred):
        reduced_scenarios[i].probability = reduced_probs[i]
        reduced_scenarios[i].id = reduced_indices[i]

    # report on scenario reduction
    print(f"Scenarios reduced in {time.time()-start:.1f}s.")
    print(f"Reduced scenarios: {reduced_indices}")
    print(f"With probabilities: {reduced_probs}")

    # reset logging setting
    if (env := settings['solver_settings'].get('env', None)): # return verbose setting
        env.setParam('OutputFlag',int(settings['solver_settings']['verbose']))
    else:
        settings['solver_settings']['log_to_console'] = settings['solver_settings']['verbose']

    return reduced_scenarios


def solve_model(
        scenarios: List[ScenarioData],
        settings: Dict,
        set_design: Dict = None
    ) -> EnergyModel:
    """Construct Scenario Programming model of system specified by settings
    with passed scenarios, and solve.

    Args:
        scenarios (List[ScenarioData]): List of Scenario objects defining
            time series data to use in Scenario Programming model.
        settings (Dict): Dictionary of settings used to construct model.
            See configs/base_settings.yaml for required keys.
        set_design (Dict, optional): Dictionary specifying system design to
            run optimization in operational mode. Defaults to None.

    Returns:
        EnergyModel: Optimised model object.
    """

    # Set up solver kwargs
    # ====================
    if (env := settings['solver_settings'].get('env', None)):
        solver_kwargs = {
            'solver_name':'gurobi',
            'env':env,
            'Threads':settings['solver_settings'].get('threads',0)
        }
    else: solver_kwargs = {'solver_name': 'highs'}
    solver_kwargs['io_api'] = 'direct' # default is 'lp', but 'direct' is faster and doesn't print to console

    if settings['solver_settings']['solver'] == 'gurobi':
        solver_options = ['TimeLimit','Crossover','CrossoverBasis','OptimalityTol']
        # see https://docs.gurobi.com/projects/optimizer/en/current/reference/parameters.html for Gurobi parameters
    else:
        solver_options = ['time_limit','log_to_console']
        # see https://ergo-code.github.io/HiGHS/dev/options/definitions for HiGHS options

    for key in solver_options:
        if key in settings['solver_settings']:
            solver_kwargs[key] = settings['solver_settings'][key]


    # Peform scenario reduction if required
    # =====================================
    if len(scenarios) > settings['probability_settings']['n_reduced_scenarios']:
        scenarios = reduce_scenarios(scenarios, settings)

    # Setup up model and solve
    # ========================
    model = EnergyModel()

    with warnings.catch_warnings():
        # filter pandas warnings, `DeprecationWarning: np.find_common_type is deprecated.`
        warnings.simplefilter("ignore", category=FutureWarning)

        # Load time series data
        for scenario in scenarios:
            scenario.load_timeseries(os.path.join(*settings['dataset_dir']))

        # Generate and solve model
        model.generate_SP(scenarios, settings['model_settings'], set_design)
        model.solve(**solver_kwargs)

    return model


def try_all_designs(
        scenarios: List[ScenarioData],
        settings: Dict,
        save_all: str = None
    ) -> EnergyModel:
    """Construct and solve Scenario Programming model of energy system
    for all combinations of available storage technologies, and select
    the best one (i.e. with lowest corrected objective).

    Args:
        scenarios (List[ScenarioData]): List of Scenario objects defining
            time series data to use in Scenario Programming model.
        settings (Dict): Dictionary of settings used to construct model.
            See configs/base_settings.yaml for required keys.
        save_all (str, optional): Path to save results for all technology
            combinations to file. Not saved to file if `None`. Defaults to None.

    Returns:
        str: Combination of storage technologies with lowest optimised cost.
    """

    available_technologies = list(settings['probability_settings']['storage'].keys())

    best_objective = float('inf')
    best_techs = None

    for techs in itertools.combinations(available_technologies, settings['model_settings']['N_technologies']):
        settings['model_settings']['storage_technologies'] = list(techs)
        techs_str = '-'.join(techs)

        if settings['solver_settings']['verbose']: print(f"Designing system with technologies: {techs_str}")

        solved_model = solve_model(scenarios, settings)

        if save_all is not None:
            solved_model.save_results(os.path.join(save_all,f'{techs_str}_design.yaml'))

        if solved_model.corrected_objective < best_objective:
            best_objective = solved_model.corrected_objective
            best_techs = techs_str

        del solved_model # free up memory

    return best_techs