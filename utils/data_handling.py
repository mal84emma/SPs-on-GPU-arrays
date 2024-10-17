"""Helper functions for handling scenario data."""

import os
import yaml
import numpy as np
import pandas as pd
import collections.abc



def update_nested_dict(d, u):
    """Update a nested dictionary with another nested dictionary.
    Solution taken from https://stackoverflow.com/a/3233356"""

    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_nested_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d


class ScenarioData:
    """Class for handling scenario data."""
    def __init__(self, cost_dict, ts_dict, storage_dict):

        self.probability = None

        self.wind_capex = cost_dict['wind_capex'] # €/kWp
        self.wind_opex = cost_dict['wind_opex'] # €/kWp/yr
        self.wind_lifetime = cost_dict['wind_lifetime'] # years
        self.solar_capex = cost_dict['solar_capex'] # €/kWp
        self.solar_opex = cost_dict['solar_opex'] # €/kWp/yr
        self.solar_lifetime = cost_dict['solar_lifetime'] # years
        self.carbon_price = cost_dict['carbon_price'] # €/kgCO2

        self.load_level = ts_dict['load_level']
        self.wind_year = ts_dict['wind_year']
        self.solar_year = ts_dict['solar_year']
        self.price_year = ts_dict['price_year']
        self.carbon_year = ts_dict['carbon_year']

        self.storage_technologies = sorted(list(storage_dict.keys()))
        self.storage_costs = {tech: storage_dict[tech]['cost'] for tech in self.storage_technologies}
        self.storage_lifetimes = {tech: storage_dict[tech]['lifetime'] for tech in self.storage_technologies}
        self.storage_efficiencies = {tech: storage_dict[tech]['efficiency'] for tech in self.storage_technologies}
        self.discharge_ratios = {tech: storage_dict[tech]['discharge_ratio'] for tech in self.storage_technologies}
        self.depths_of_discharge = {tech: storage_dict[tech]['depth_of_discharge'] for tech in self.storage_technologies}

    def __str__(self):

        ts_dict, cost_dict, storage_dict = self.to_file(None, save=False)

        str = ''
        str += 'Scenario Data\n'
        str += '=============\n'

        str += 'Time Series Data\n'
        str += '----------------\n'
        for key, val in ts_dict.items():
            str += f'{key}: {val}\n'

        str += '\nCost Data\n'
        str += '---------\n'
        for key, val in cost_dict.items():
            str += f'{key}: {val}\n'

        str += '\nStorage Data\n'
        str += '------------\n'
        for tech, data in storage_dict.items():
            str += f'{tech}:\n'
            for key, val in data.items():
                str += f'  {key}: {val}\n'

        return str

    def to_file(self, fpath, save=True):
        """Save scenario info to json."""

        cost_dict = {
            'wind_capex': self.wind_capex,
            'wind_opex': self.wind_opex,
            'wind_lifetime': self.wind_lifetime,
            'solar_capex': self.solar_capex,
            'solar_opex': self.solar_opex,
            'solar_lifetime': self.solar_lifetime,
            'carbon_price': self.carbon_price
        }
        ts_dict = {
            'load_level': self.load_level,
            'wind_year': self.wind_year,
            'solar_year': self.solar_year,
            'price_year': self.price_year,
            'carbon_year': self.carbon_year
        }
        storage_dict = {
            tech: {
                'cost': self.storage_costs[tech],
                'lifetime': self.storage_lifetimes[tech],
                'annual_cost': self.storage_costs[tech] / self.storage_lifetimes[tech],
                'efficiency': self.storage_efficiencies[tech],
                'discharge_ratio': self.discharge_ratios[tech],
                'depth_of_discharge': self.depths_of_discharge[tech]
            } for tech in self.storage_technologies
        }

        if save:
            if os.path.dirname(fpath) != '':
                if not os.path.exists(os.path.dirname(fpath)):
                    os.makedirs(os.path.dirname(fpath))

            with open(fpath, 'w') as f:
                yaml.dump({
                    'cost_values': cost_dict,
                    'timeseries_values': ts_dict,
                    'storage_values': storage_dict
                }, f, sort_keys=False)

        return cost_dict, ts_dict, storage_dict

    def from_file(fpath):

        with open(fpath, 'r') as f: data = yaml.safe_load(f)

        return ScenarioData(data['cost_values'], data['timeseries_values'], data['storage_values'])

    def load_timeseries(self, dataset_dir):
        """Load timeseries data from CSV files."""

        self.load = np.ones(8760) * self.load_level
        self.norm_wind_gen = pd.read_csv(os.path.join(dataset_dir,'wind',f'{self.wind_year}.csv'))['Wind generation [kW/kWp]'].to_numpy()
        self.norm_solar_gen = pd.read_csv(os.path.join(dataset_dir,'solar',f'{self.solar_year}.csv'))['Solar generation [kW/kWp]'].to_numpy()
        self.elec_prices = pd.read_csv(os.path.join(dataset_dir,'price',f'{self.price_year}.csv'))['Electricity price [EUR/kWh]'].to_numpy()
        self.carbon_intensity = pd.read_csv(os.path.join(dataset_dir,'carbon',f'{self.carbon_year}.csv'))['Carbon intensity [kgCO2/kWh]'].to_numpy()
