"""Helper functions for working with models."""

import os
import warnings
import itertools

from typing import List, Dict
from utils import ScenarioData
from energy_model import EnergyModel



def solve_model(
        scenarios: List[ScenarioData],
        settings: Dict
    ) -> EnergyModel:
    """Construct Scenario Programming model of system specified by settings
    with passed scenarios, and solve.

    Args:
        scenarios (List[ScenarioData]): List of Scenario objects defining
            time series data to use in Scenario Programming model.
        settings (Dict): Dictionary of settings used to construct model.
            See configs/base_settings.yaml for required keys.

    Returns:
        EnergyModel: Optimised model object.
    """

    # Set up solver kwargs
    env = settings['solver_settings'].get('env', None)
    if env is not None:
        solver_kwargs = {
            'solver_name':'gurobi',
            'env':env,
            'Threads':settings['solver_settings'].get('threads',0)
        }
    else: solver_kwargs = {'solver_name': 'highs'}
    solver_kwargs['io_api'] = 'direct' # default is 'lp', but 'direct' is faster and doesn't print to console

    # Setup up model and solve
    model = EnergyModel()

    with warnings.catch_warnings():
        # filter pandas warnings, `DeprecationWarning: np.find_common_type is deprecated.`
        warnings.simplefilter("ignore", category=FutureWarning)

        for scenario in scenarios:
            scenario.load_timeseries(os.path.join(*settings['dataset_dir']))

        model.generate_SP(scenarios, settings['model_settings'])
        model.solve(**solver_kwargs)

    return model


def try_all_designs(
        scenarios: List[ScenarioData],
        settings: Dict,
        save_all: bool = False
    ) -> EnergyModel:
    """Construct and solve Scenario Programming model of energy system
    for all combinations of available storage technologies, and select
    the best one (i.e. with lowest corrected objective).

    Args:
        scenarios (List[ScenarioData]): List of Scenario objects defining
            time series data to use in Scenario Programming model.
        settings (Dict): Dictionary of settings used to construct model.
            See configs/base_settings.yaml for required keys.
        save_all (bool, optional): Whether to save designs with all storage
            combinations to file. Defaults to False.

    Returns:
        EnergyModel: Optimised model object with best storage combination.
    """

    available_technologies = list(settings['probability_settings']['storage'].keys())

    best_objective = float('inf')
    best_model = None

    for techs in itertools.combinations(available_technologies, settings['model_settings']['N_technologies']):
        settings['model_settings']['storage_technologies'] = list(techs)
        techs_str = '-'.join(techs)

        if settings['solver_settings']['verbose']: print(f"Designing system with technologies: {techs_str}")

        solved_model = solve_model(scenarios, settings)

        if save_all: solved_model.save_results(os.path.join(*settings['results_dir'],'prior',f'{techs_str}_design.yaml'))
        if solved_model.corrected_objective < best_objective:
            best_objective = solved_model.corrected_objective
            best_model = solved_model

    return best_model