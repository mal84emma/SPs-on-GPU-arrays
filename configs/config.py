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