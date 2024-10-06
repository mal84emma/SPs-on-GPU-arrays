"""Helper functions for working with models."""

import os
import warnings

from energy_model import EnergyModel


def solve_model(scenarios, settings):

    # Set up solver kwargs
    env = settings['solver_settings'].get('env',None)
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

        model.generate_SP(scenarios,settings['model_settings'])
        model.solve(**solver_kwargs)

    return model