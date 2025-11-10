"""Solve SP distributed across GPUs."""

import csv
import os
import sys
import json
import time
import numpy as np
from scipy import optimize
from tqdm import tqdm

from model.model_handling import solve_model
from model.utils import get_experiment_config, ScenarioData

SOLAR_CAPACITY = 500000  # fix solar capacity for all designs


def build_design(capacities):
    """Build design dict from capacities."""

    design = {
        "wind_capacity": {"unit": "kW", "value": capacities[0]},
        "solar_capacity": {"unit": "kWp", "value": SOLAR_CAPACITY},
        "storage_technologies": ["Li-ion"],
        "storage_capacities": {"Li-ion": {"unit": "kWh", "value": capacities[1]}},
    }

    return design


def build_meta_constraints(scenarios, settings, base_params):
    """Build constraints for meta-optimisation."""

    max_storage_cost = np.max(
        [s.storage_costs["Li-ion"] / s.storage_lifetimes["Li-ion"] for s in scenarios]
    )

    solar_cost = (
        base_params["cost_values"]["solar_capex"]
        / base_params["cost_values"]["solar_lifetime"]
        + base_params["cost_values"]["solar_opex"]
    )
    wind_cost = (
        base_params["cost_values"]["wind_capex"]
        / base_params["cost_values"]["wind_lifetime"]
        + base_params["cost_values"]["wind_opex"]
    )

    remaining_budget = (
        settings["model_settings"]["capex_budget"] - solar_cost * SOLAR_CAPACITY
    )

    constr_A = np.array(
        [
            [wind_cost, 0],
            [0, max_storage_cost],
            [wind_cost, max_storage_cost],
        ]
    )
    lb = np.array([0, 0, 0])
    ub = np.array([remaining_budget, remaining_budget, remaining_budget])

    return optimize.LinearConstraint(constr_A, lb, ub)


def pre_solve(scenarios, settings):
    """Pre-solve each scenario individually to get mean solution as initial guess."""

    wind_caps = []
    storage_caps = []

    for scenario in tqdm(scenarios):
        single_solution = solve_model([scenario], settings)
        wind_caps.append(
            single_solution.model.variables["wind_capacity"].solution.values
        )
        storage_caps.append(
            single_solution.model.variables["Li-ion_capacity"].solution.values
        )

    # need to add in a buffer (try 20%) to ensure feasibility
    return np.array([np.mean(wind_caps) * 0.8, np.mean(storage_caps) * 0.8])


def solve_step_individual(design, scenario, settings):
    """Solve single scenario model for given design."""

    solved_model = solve_model([scenario], settings, design)
    obj = solved_model.corrected_objective

    # get gradient estimates from dual values
    wind_grad = solved_model.model.constraints["wind_capacity"].dual.values
    storage_grad = solved_model.model.constraints["Li-ion_capacity"].dual.values
    grad = np.array([wind_grad, storage_grad])

    return obj, grad


def solve_step_distributed(capacities, scenarios, settings, out_dir):
    """Solve scenario optimisations distributed across devices for
    given design/capacities and return mean cost and gradient."""

    print(f"Testing capacities: wind {capacities[0]}, storage {capacities[1]}")

    test_design = build_design(capacities)

    costs = []
    grads = []

    # optimise each scenario separately
    times = []
    for scenario in tqdm(scenarios):
        start = time.time()

        obj, grad = solve_step_individual(test_design, scenario, settings)

        end = time.time()
        times.append(end - start)

        print(grad)
        costs.append(obj)
        grads.append(grad)

    with open(os.path.join(out_dir, "solve_times.csv"), "a") as f:
        writer = csv.writer(f)
        writer.writerow(times)

    print(f"Objective: {np.mean(costs)}, Gradient: {np.mean(grads, axis=0)}")

    # return mean cost and gradient over scenarios
    return np.mean(costs), np.mean(grads, axis=0)


if __name__ == "__main__":
    # get no. of scenarios from command line
    if len(sys.argv) > 1:
        num_scenarios = int(sys.argv[1])
    else:
        num_scenarios = 1

    # load config
    settings, base_params = get_experiment_config("distributed")
    prob_settings = settings["probability_settings"]

    save_dir = os.path.join(*settings["results_dir"])

    # load scenarios
    scenarios_dir = os.path.join(*settings["scenarios_dir"])
    scenarios = [
        ScenarioData.from_file(os.path.join(scenarios_dir, f"scenario_{i}.yaml"))
        for i in range(settings["probability_settings"]["n_prior_samples"])
    ]

    scenarios = scenarios[:num_scenarios]

    # create output dir
    out_dir = os.path.join(save_dir, f"s{num_scenarios}")
    os.makedirs(out_dir, exist_ok=True)

    # exact solve
    print("Solving exact model for comparison...")
    start = time.time()

    exact_soln = solve_model(scenarios, settings)

    end = time.time()
    print("Exact solution:")
    exact_obj = exact_soln.corrected_objective
    exact_wind_cap = exact_soln.model.variables["wind_capacity"].solution.values
    exact_storage_cap = exact_soln.model.variables["Li-ion_capacity"].solution.values

    exact_results = {
        "objective": float(exact_obj),
        "wind_capacity": float(exact_wind_cap),
        "storage_capacity": float(exact_storage_cap),
        "solve_time": end - start,
    }

    with open(os.path.join(out_dir, "exact.json"), "w") as f:
        json.dump(exact_results, f, indent=4)

    for key, value in exact_results.items():
        print(f"{key}: {value}")
    print("")

    # build meta constraint - handling these needs some overcoming
    # scipy doesn't have a nice solver for constrained problems with gradients
    # I'm not sure if there is a nice solver for PWL problems with constraints
    constraints = build_meta_constraints(scenarios, settings, base_params)

    # pre-solve scenario individually for good initial guess
    print("Pre-solving...")
    initial_guess = pre_solve(scenarios, settings)
    print(f"Initial guess: {initial_guess}")

    # optimise distributedly
    print("Solving distributed...")
    result = optimize.minimize(
        solve_step_distributed,
        x0=initial_guess,
        args=(scenarios, settings, out_dir),
        method="trust-constr",
        jac=True,
        options={"disp": True, "xtol": 250, "gtol": 1e-2, "maxiter": 100},
        constraints=[constraints],
    )

    # save results
    with open(os.path.join(out_dir, "solve_times.csv"), "r") as f:
        # load csv to np.array
        reader = csv.reader(f)
        times = np.array([[float(t) for t in row] for row in reader])
        max_times = np.max(times, axis=1)
        total_time = np.sum(max_times)
    # NOTE: the solve time stated by cuPDLPx is much smaller because this
    # doesn't account for problem setup & data transfer times. But a custom
    # implemenation of this method could get rid of these overheads.

    distributed_results = {
        "objective": float(result.fun),
        "wind_capacity": float(result.x[0]),
        "storage_capacity": float(result.x[1]),
        "solve_time": total_time,
    }

    with open(os.path.join(out_dir, "distributed.json"), "w") as f:
        json.dump(distributed_results, f, indent=4)

    # report results
    print(result)
    print("\n=== Distributed ===")
    for key, value in distributed_results.items():
        print(f"{key}: {value}")

    print("\n=== Exact ===")
    for key, value in exact_results.items():
        print(f"{key}: {value}")
