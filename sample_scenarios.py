# Perform scenario sampling from prior distribution and save to file

import os
import sys
import time
import numpy as np

from model.utils import get_experiment_config
from model.prob_models import prior_model


if __name__ == "__main__":
    expt_name = str(sys.argv[1])
    settings, base_params = get_experiment_config(expt_name)
    prob_settings = settings["probability_settings"]

    # ========================================

    start = time.time()

    # Sample from prior
    np.random.seed(42)  # for reproducibility
    save_dir = os.path.join(*settings["scenarios_dir"])

    scenarios = prior_model(
        prob_settings,
        base_params["cost_values"],
        base_params["timeseries_values"],
        base_params["storage_values"],
        n_samples=prob_settings["n_prior_samples"],
    )

    for i in range(prob_settings["n_prior_samples"]):
        scenarios[i].to_file(os.path.join(save_dir, "thetas", f"scenario_{i}.yaml"))

    print(f"Scenarios sampled in {time.time() - start:.1f}s")
