"""Definition of probabilistic models for decision problem."""

import os
import numpy as np
from scipy.stats import truncnorm
from cmdstanpy import CmdStanModel

# Turn off Stan logging
import logging
logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.WARNING)

from utils.data_handling import ScenarioData


def truncnorm_sample(mu, sigma, lower=-2, upper=2):
    """Sample from a truncated normal distribution.
    Default with \pm 2 standard deviations."""
    return truncnorm.rvs(lower, upper, loc=mu, scale=sigma)


def prior_model(prob_settings, base_cost_dict, base_ts_dict, base_storage_dict, n_samples=64):
    """TODO: perform sampling and return list of ScenarioData objects for
    thetas and zs."""

    assert all([key in prob_settings.keys() for key in base_storage_dict.keys()]), "Must provide probability settings for all storage technologies."

    theta_scenarios = []
    z_scenarios = []

    ## Perform theta sampling
    for i in range(n_samples):
        ts_dict = base_ts_dict.copy()
        storage_dict = base_storage_dict.copy()

        # Sample timeseries parameters
        ts_dict['load_level'] = truncnorm_sample(*prob_settings['load_level'])
        ts_dict['wind_year'] = np.random.choice(prob_settings['wind_years'])
        ts_dict['solar_year'] = np.random.choice(prob_settings['solar_years'])

        # Sample storage parameters (all with truncnorm distributions)
        for tech in storage_dict.keys():
            for key in ['cost', 'lifetime', 'efficiency']:
                storage_dict[tech][key] = truncnorm_sample(*prob_settings[tech][key])

        theta_scenarios.append(ScenarioData(base_cost_dict, ts_dict, storage_dict))

    ## Perform z sampling (storage only)
    for i in range(n_samples):
        ts_dict, cost_dict, storage_dict = theta_scenarios[i].to_file(None, save=False)

        for tech in storage_dict.keys():
            for key in ['cost', 'lifetime', 'efficiency']:
                # TODO: figure out measurement probability model
                storage_dict[tech][key] = ...(*prob_settings[tech][key])

        z_scenarios.append(ScenarioData(ts_dict, cost_dict, storage_dict))

    return theta_scenarios, z_scenarios


def posterior_model(z_scenario, prob_settings, n_samples=64):
    """TODO: take in scenario defining measurement values, sample from posterior,
    and return list of ScenarioData objects for varthetas."""

    ts_dict, cost_dict, storage_dict = z_scenario.to_file(None, save=False)

    ## Perform vartheta sampling (storage only)
    vartheta_samples = {}
    for tech in storage_dict.keys():
        for key in ['cost', 'lifetime', 'efficiency']:
            # TODO: implement STAN posterior sampling
            samples = ...
            vartheta_samples[tech][key] = samples

    ## Assign samples to scenarios and create objects
    vartheta_scenarios = []
    for i in range(n_samples):
        for tech in storage_dict.keys():
            for key in ['cost', 'lifetime', 'efficiency']:
                storage_dict[tech][key] = vartheta_samples[tech][key][i]
        vartheta_scenarios.append(ScenarioData(ts_dict, cost_dict, storage_dict))

    return vartheta_scenarios