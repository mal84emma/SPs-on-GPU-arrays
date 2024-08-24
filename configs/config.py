"""Common configuration for experiments."""

import os

# Directories and file patterns
dataset_dir = os.path.join('data','processed')
results_dir = os.path.join('experiments','results')

# Available years of data
wind_years = list(range(2010,2019))
solar_years = list(range(2010,2019))
price_years = list(range(2010,2019))
carbon_years = list(range(2010,2019))

# Location of assets
site_location = []
wind_location = []