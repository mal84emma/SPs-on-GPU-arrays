"""Benchmark solver run times for a single scenario optimisation."""

import os

from model.model_handling import solve_model
from model.utils import get_experiment_config, ScenarioData

if __name__ == "__main__":
    # load config
    settings, base_params = get_experiment_config("base")
    prob_settings = settings["probability_settings"]

    save_dir = os.path.join(*settings["results_dir"], "benchmark")

    # load scenarios
    scenarios_dir = os.path.join(*settings["scenarios_dir"])
    scenarios = [
        ScenarioData.from_file(os.path.join(scenarios_dir, f"scenario_{i}.yaml"))
        for i in range(100)
    ]

    # solve & save
    for solver in ["highs", "cupdlpx"]:
        print(f"Solving with {solver}...")
        settings["solver_settings"]["solver_name"] = solver

        # optimise
        solved_model = solve_model(scenarios, settings)

        solved_model.save_results(os.path.join(save_dir, f"{solver}_test.yaml"))
