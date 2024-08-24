"""Helper functions for loading and saving scenarios."""

import csv
import ast
import numpy as np


def format_scenario_tuple(building_tuple):
    """Format first two arugments of building scenario tuples to be integers."""
    return tuple([int(t) if i < 2 else t for i,t in enumerate(building_tuple)])

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

def save_eval_results(results, design, scenarios, out_path):
    """Save evaluation results for multiple scenarios to CSV."""

    design_header = ['Parameter', 'Units']
    design_header.extend([f'SB{i}' for i in range(scenarios.shape[1])])
    design_rows = [
        ['Battery Capacity', 'kWh', *design['battery_capacities'].flatten()],
        ['Solar Capacity', 'kWp', *design['solar_capacities'].flatten()],
        ['Grid Con. Capacity', 'kW', design['grid_con_capacity']]
    ]

    evals_header = ['Scenario no.','Total cost','Elec. price','Carbon cost','Grid excess cost','Grid capacity cost','Battery cost','Solar cost']
    evals_header.extend([f'SB{i}' for i in range(scenarios.shape[1])])

    with open(out_path, 'w') as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(['Design'])
        writer.writerow(design_header)
        for row in design_rows:
            writer.writerow(row)

        writer.writerow(['Evaluations'])
        writer.writerow(evals_header)
        for i, (result,scenario) in enumerate(zip(results,scenarios)):
            writer.writerow([i,result['objective'],*result['objective_contrs'],*[format_scenario_tuple(tuple(bs)) for bs in scenario]])

def load_eval_results(in_path):
    """Load evaluation results for multiple scenarios from CSV."""

    results = []
    with open(in_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        eval_rows = [row for row in reader][7:]

    for row in eval_rows:
        result = {
            'objective': float(row[1]),
            'objective_contrs': [float(t) for t in row[2:8]]
        }
        results.append(result)

    return results