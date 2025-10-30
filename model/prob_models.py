"""Definition of probabilistic models for decision problem."""

import copy
import numpy as np
from scipy.stats import norm, truncnorm
from typing import List

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
                n_samples:int=64) -> List[ScenarioData]:
    """Sample scenarios from prior distribution. Output as lists of ScenarioData objects

    Args:
        prob_settings (dict): Dictionary of probability settings defining distributions.
            (`probability_settings` from `settings.yaml`)
        base_cost_dict (dict): Template dictionary of cost values.
        base_ts_dict (dict): Template dictionary of time series values.
        base_storage_dict (dict): Template dictionary of storage values.
        n_samples (int, optional): Number of scenarios to draw from prior. Defaults to 64.

    Returns:
        List[ScenarioData]: Lists of sampled scenarios.
    """

    assert all([key in prob_settings['storage'].keys() for key in base_storage_dict.keys()]), "Must provide probability settings for all storage technologies."

    scenarios = []

    for _ in range(n_samples):
        ts_dict = copy.deepcopy(base_ts_dict)
        storage_dict = copy.deepcopy(base_storage_dict)

        # Sample timeseries parameters
        ts_dict['load_level'] = float(truncnorm_sample(*prob_settings['load_level']))
        ts_dict['wind_year'] = int(np.random.choice(prob_settings['wind_years']))
        ts_dict['solar_year'] = int(np.random.choice(prob_settings['solar_years']))
        ts_dict['price_year'] = prob_settings['price_year']
        ts_dict['carbon_year'] = prob_settings['carbon_year']

        # Sample storage parameters (all with truncnorm distributions)
        for tech in storage_dict.keys():
            for key in ['cost', 'lifetime', 'efficiency']:
                storage_dict[tech][key] = float(truncnorm_sample(*prob_settings['storage'][tech][key]))

        scenarios.append(ScenarioData(base_cost_dict, ts_dict, storage_dict))

    return scenarios
