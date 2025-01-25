"""Definition of probabilistic models for decision problem."""

import os
import yaml
import copy
import numpy as np
from scipy.stats import norm, truncnorm
from cmdstanpy import CmdStanModel
from typing import Tuple, Iterable

# Turn off Stan logging
import logging
logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.WARNING)

from utils.data_handling import ScenarioData


def truncnorm_sample(mu:float, sigma:float, lower=-2.0, upper=2.0) -> np.array:
    """Sample from a truncated normal distribution.
    Default truncation is \pm 2 standard deviations.

    Args:
        mu (float): Mean of the distribution.
        sigma (float): Standard deviation of the distribution.
        lower (float, optional): Lower cutoff (no. of sigma). Defaults to -2.0.
        upper (float, optional): Upper cutoff (no. of sigma). Defaults to 2.0.

    Returns:
        np.array: Samples of random variable.
    """
    return truncnorm.rvs(lower, upper, loc=mu, scale=sigma)


def prior_model(prob_settings:dict, base_cost_dict:dict, base_ts_dict:dict, base_storage_dict:dict,
                n_samples:int=64) -> Tuple[Iterable[ScenarioData], Iterable[ScenarioData]]:
    """Sample thetas and zs from prior distribution. Output as lists of ScenarioData objects

    Args:
        prob_settings (dict): Dictionary of probability settings defining distributions.
            (`probability_settings` from `settings.yaml`)
        base_cost_dict (dict): Template dictionary of cost values.
        base_ts_dict (dict): Template dictionary of time series values.
        base_storage_dict (dict): Template dictionary of storage values.
        n_samples (int, optional): Number of scenarios to draw from prior. Defaults to 64.

    Returns:
        Tuple[Iterable[ScenarioData], Iterable[ScenarioData]]: Lists of theta and z scenarios.
    """

    assert all([key in prob_settings['storage'].keys() for key in base_storage_dict.keys()]), "Must provide probability settings for all storage technologies."

    theta_scenarios = []
    z_scenarios = []

    ## Perform theta sampling
    for i in range(n_samples):
        ts_dict = copy.deepcopy(base_ts_dict)
        storage_dict = copy.deepcopy(base_storage_dict)

        # Sample timeseries parameters
        ts_dict['load_level'] = float(truncnorm_sample(*prob_settings['load_level']))
        ts_dict['wind_year'] = int(np.random.choice(prob_settings['wind_years']))
        ts_dict['solar_year'] = int(np.random.choice(prob_settings['solar_years']))

        # Sample storage parameters (all with truncnorm distributions)
        for tech in storage_dict.keys():
            for key in ['cost', 'lifetime', 'efficiency']:
                storage_dict[tech][key] = float(truncnorm_sample(*prob_settings['storage'][tech][key]))

        theta_scenarios.append(ScenarioData(base_cost_dict, ts_dict, storage_dict))

    ## Perform z sampling (storage only)
    for i in range(n_samples):
        cost_dict, ts_dict, storage_dict = theta_scenarios[i].to_file(None, save=False)

        for tech in storage_dict.keys():
            for key in ['cost', 'lifetime', 'efficiency']:
                storage_dict[tech][key] = float(norm.rvs(
                    loc=storage_dict[tech][key], # theta sample
                    scale=prob_settings['storage'][tech][key][1]*prob_settings['measurement_sigma_reduction'] # reduced sigma
                ))

        z_scenarios.append(ScenarioData(cost_dict, ts_dict, storage_dict))

    return theta_scenarios, z_scenarios


def posterior_model(z_scenario:ScenarioData, prob_settings:dict,
                    n_samples:int=64) -> Iterable[ScenarioData]:
    """Sample vartheta scenarios from posterior distribution given a scenario z.

    Args:
        z_scenario (ScenarioData): Measured scenario to condition on.
        prob_settings (dict): Dictionary of probability settings defining distribution.
        n_samples (int, optional): Number of scenarios to draw from posterior. Defaults to 64.

    Returns:
        Iterable[ScenarioData]: List of vartheta scenarios.
    """

    cost_dict, ts_dict, storage_dict = z_scenario.to_file(None, save=False)

    ## Sample timeseries parameters (unmeasured) using prior
    thetas,_ = prior_model(prob_settings, cost_dict, ts_dict, storage_dict, n_samples)

    ## Perform vartheta sampling (storage only)
    vartheta_samples = {}
    for tech in storage_dict.keys():
        vartheta_samples[tech] = {}
        for key in ['cost', 'lifetime', 'efficiency']:
            # Load STAN model and perform sampling
            posterior_file = os.path.join('stan_models','posterior.stan')
            stan_model = CmdStanModel(stan_file=posterior_file)

            data = {
                'mu':prob_settings['storage'][tech][key][0],
                'sigma':prob_settings['storage'][tech][key][1],
                'reduction_factor':prob_settings['measurement_sigma_reduction'],
                'z':storage_dict[tech][key]
            }
            inits = {'theta':prob_settings['storage'][tech][key][0]}

            posterior_fit = stan_model.sample(
                    data=data,
                    inits=inits,
                    iter_warmup=n_samples,
                    iter_sampling=n_samples*prob_settings['sampling_thin_factor'],
                    chains=1,
                    show_progress=False
                )

            vartheta_samples[tech][key] = posterior_fit.stan_variable('theta')[::prob_settings['sampling_thin_factor']]

    ## Assign samples to scenarios and create objects
    vartheta_scenarios = []
    for i in range(n_samples):
        _,ts_dict_i,_ = thetas[i].to_file(None, save=False) # resampled ts params
        for tech in storage_dict.keys():
            for key in ['cost', 'lifetime', 'efficiency']:
                storage_dict[tech][key] = float(vartheta_samples[tech][key][i])
        vartheta_scenarios.append(ScenarioData(cost_dict, ts_dict_i, storage_dict))

    return vartheta_scenarios


if __name__ == '__main__':

    n_samples = 5

    with open(os.path.join('configs','base_settings.yaml'), 'r') as f: settings = yaml.safe_load(f)
    prob_settings = settings['probability_settings']

    with open(os.path.join('configs','base_params.yaml'), 'r') as f: params = yaml.safe_load(f)
    cost_dict = params['cost_values']
    ts_dict = params['timeseries_values']
    storage_dict = params['storage_values']

    # Test prior sampling
    print('Testing prior sampling...')
    theta_scenarios, z_scenarios = prior_model(prob_settings, cost_dict, ts_dict, storage_dict, n_samples=n_samples)
    for i in range(n_samples):
        print(f'Sample {i}')
        for attr in ['storage_costs','storage_lifetimes','storage_efficiencies']:
            print(f'{attr}, t', getattr(theta_scenarios[i],attr))
            print(f'{attr}, z', getattr(z_scenarios[i],attr))
    print('')

    # Test posterior sampling
    print('Testing posterior sampling...')
    vartheta_scenarios = posterior_model(z_scenarios[0], prob_settings, n_samples=n_samples)
    for attr in ['storage_costs','storage_lifetimes','storage_efficiencies']:
        print(attr)
        print(f'original', getattr(z_scenarios[0],attr))
        for i in range(n_samples):
            print(f'sample {i}', getattr(vartheta_scenarios[i],attr))