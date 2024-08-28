"""Common configuration for experiments."""

import os

# Directories and file patterns
dataset_dir = os.path.join('data','processed')
results_dir = os.path.join('experiments','results')

# Available years of data
wind_years = list(range(2010,2020))
solar_years = list(range(2010,2020))
price_years = [2023]
carbon_years = [2023]

# Location of assets
site_location = [51.95,4.1]
wind_location = [52.7,3.5]

# Default energy model settings
model_settings = {
    'T':8760,
    'initial_SoC':0.5,
    'max_storage_cap':1e9, # slack bound
    'N_technologies': 1,
    'allow_elec_purchase': True,
    'grid_capacity': 500e3, # kW
    'capex_budget': 10e9/20, # €/yr
    'solar_capacity_limit': 500e3, # kWp
    #'technologies_to_use': ['li-ion'],
}

# Probabilistic model settings
prob_settings = { # tuples are [mu,sigma]
    'TODO': ...,
    'measurement_sigma_reduction': 0.25, # TODO: confirm
    'load_level': [250000,25000],
    'wind_years': wind_years,
    'solar_years': solar_years,
    'li-ion': { # TODO: work out distributions
        'cost': [],
        'lifetime': [],
        'efficiency': [],
    },
}