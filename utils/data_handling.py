"""Helper functions for handling scenario data."""

import os
import csv
import ast
import numpy as np
import pandas as pd


class ScenarioData:
    """Class for handling scenario data."""
    def __init__(self, ts_dict, cost_dict, storage_dict):

        self.probability = None

        self.load_level = ts_dict['load_level']
        self.wind_year = ts_dict['wind_year']
        self.solar_year = ts_dict['solar_year']
        self.price_year = ts_dict['price_year']
        self.carbon_year = ts_dict['carbon_year']

        self.wind_capex = cost_dict['wind_capex'] # €/kWp
        self.wind_opex = cost_dict['wind_opex'] # €/kWp/yr
        self.wind_lifetime = cost_dict['wind_lifetime'] # years
        self.solar_capex = cost_dict['solar_capex'] # €/kWp
        self.solar_opex = cost_dict['solar_opex'] # €/kWp/yr
        self.solar_lifetime = cost_dict['solar_lifetime'] # years
        self.carbon_price = cost_dict['carbon_price'] # €/kgCO2

        self.storage_technologies = sorted(list(storage_dict.keys()))
        self.storage_costs = [storage_dict[tech]['cost'] for tech in self.storage_technologies]
        self.storage_lifetimes = [storage_dict[tech]['lifetime'] for tech in self.storage_technologies]
        self.storage_efficiencies = [storage_dict[tech]['efficiency'] for tech in self.storage_technologies]
        self.discharge_ratios = [storage_dict[tech]['discharge_ratio'] for tech in self.storage_technologies]

    def load_timeseries(self, dataset_dir):
        """Load timeseries data from CSV files."""

        self.load = np.ones(8760) * self.load_level
        self.norm_wind_gen = pd.read_csv(os.path.join(dataset_dir,'wind',f'{self.wind_year}.csv'))['Wind generation [kW/kWp]'].to_numpy()
        self.norm_solar_gen = pd.read_csv(os.path.join(dataset_dir,'solar',f'{self.solar_year}.csv'))['Solar generation [kW/kWp]'].to_numpy()
        self.elec_prices = pd.read_csv(os.path.join(dataset_dir,'price',f'{self.price_year}.csv'))['Electricity price [EUR/kWh]'].to_numpy()
        self.carbon_intensity = pd.read_csv(os.path.join(dataset_dir,'carbon',f'{self.carbon_year}.csv'))['Carbon intensity [kgCO2/kWh]'].to_numpy()



def save_scenarios(scenarios, measurements, out_path):
    """Save sampled scenarios to CSV."""

    header = ['Scenario no.']
    header.extend([f'SB{i}' for i in range(scenarios.shape[1])])
    header.extend([f'BM{i}' for i in range(measurements.shape[1])])

    with open(out_path, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for scenario_no, scenario in enumerate(scenarios):
            row = [scenario_no]
            row += [format_scenario_tuple(tuple(bs)) for bs in scenario]
            row += [format_scenario_tuple(tuple(bm)) for bm in measurements[scenario_no]]
            writer.writerow(row)

def load_scenarios(in_path):
    """Load sampled scenarios from CSV."""

    with open(in_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        n_buildings = (len(header) - 1) // 2

        scenarios = []
        measurements = []
        for row in reader:
            scenarios.append([ast.literal_eval(t) for t in row[1:n_buildings+1]])
            measurements.append([ast.literal_eval(t) for t in row[n_buildings+1:]])

    return np.array(scenarios), np.array(measurements)

def save_design_results(results, out_path):
    """Save LP design results & used scenarios to CSV."""

    design_header = ['Parameter', 'Units']
    design_header.extend([f'SB{i}' for i in range(results['reduced_scenarios'].shape[1])])
    design_rows = [
        ['Battery Capacity', 'kWh', *results['battery_capacities'].flatten()],
        ['Solar Capacity', 'kWp', *results['solar_capacities'].flatten()],
        ['Grid Con. Capacity', 'kW', results['grid_con_capacity']]
    ]

    objective_header = ['Objective Components', 'Value ($)']
    obj_contr_labels = ['Total','Elec. Price','Carbon Cost','Grid Ex. Cost','Grid Cap. Cost','Battery Cost','Solar Cost']
    obj_contrs = [results['objective'],*results['objective_contrs']]

    scenarios_header = ['Scenario no.', 'Prob']
    scenarios_header.extend([f'SB{i}' for i in range(results['reduced_scenarios'].shape[1])])

    with open(out_path, 'w') as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(['Design'])
        writer.writerow(design_header)
        for row in design_rows:
            writer.writerow(row)

        writer.writerow(['Objective'])
        writer.writerow(objective_header)
        for label,val in zip(obj_contr_labels,obj_contrs):
            writer.writerow([label,val])

        writer.writerow(['Reduced Scenarios'])
        writer.writerow(scenarios_header)
        for scenario_no, (scenario,prob) in enumerate(zip(results['reduced_scenarios'],results['reduced_probs'])):
            writer.writerow([scenario_no] + [prob] + [format_scenario_tuple(tuple(bs)) for bs in scenario])

def load_design_results(in_path):
    """Load LP design results & used scenarios from CSV."""

    results = {}
    with open(in_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]

    # Load design results.
    results['battery_capacities'] = np.array([float(t) for t in rows[2][2:]])[np.newaxis].T
    results['solar_capacities'] = np.array([float(t) for t in rows[3][2:]])[np.newaxis].T
    results['grid_con_capacity'] = float(rows[4][2])

    # Load objective results.
    results['objective'] = float(rows[7][1])
    results['objective_contrs'] = np.array([float(t) for t in [row[1] for row in rows[8:14]]])

    # Load reduced scenarios.
    results['reduced_scenarios'] = np.array([[ast.literal_eval(t) for t in row[2:]] for row in rows[16:]])
    results['reduced_probs'] = np.array([float(row[1]) for row in rows[16:]])

    return results