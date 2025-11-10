"""Plot results."""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":
    # load results
    # ==========================
    results_dir_pattern = os.path.join("results", "distributed", "s{n}")

    scenario_nums = [2, 5, 10, 20, 50, 100, 150]

    unified_times = []
    distributed_times = []
    distributed_iters = []

    obj_diffs = []
    wind_diffs = []
    storage_diffs = []

    for s in scenario_nums:
        results_dir = results_dir_pattern.format(n=s)

        # unified times
        with open(os.path.join(results_dir, "exact.json"), "r") as f:
            exact_data = json.load(f)
            unified_times.append(exact_data["solve_time"])

        # distributed times
        with open(os.path.join(results_dir, "distributed.json"), "r") as f:
            distributed_data = json.load(f)
            distributed_times.append(distributed_data["solve_time"])

        # distributed iters
        with open(os.path.join(results_dir, "solve_times.csv"), "r") as f:
            # number of lines in csv is number of meta solver iterations
            distributed_iters.append(sum(1 for _ in f))

        # solution accuracy
        obj_diffs.append(
            (exact_data["objective"] - distributed_data["objective"])
            / exact_data["objective"]
        )
        wind_diffs.append(
            (exact_data["wind_capacity"] - distributed_data["wind_capacity"])
            / exact_data["wind_capacity"]
        )
        storage_diffs.append(
            (exact_data["storage_capacity"] - distributed_data["storage_capacity"])
            / exact_data["storage_capacity"]
        )

    # compute estimated solve times
    corrected_times = np.array(distributed_iters) * 6
    # average time per iteration is ~6s prior to memory leak hitting VRAM limits
    estimated_times = np.array(distributed_iters) * 1
    # duPDLPx solve time is 0.6-0.75s for individual scenario, but add some buffer

    # plot solve time
    # ==========================
    plt.figure()
    plt.plot(
        scenario_nums,
        unified_times,
        label="Unified",
        c="b",
        marker="o",
    )
    plt.plot(
        scenario_nums,
        distributed_times,
        label="Distributed",
        c="r",
        marker="o",
    )
    plt.plot(
        scenario_nums,
        corrected_times,
        label="Corrected",
        c="r",
        marker="o",
        alpha=0.5,
        linestyle=":",
    )  # estimate of solve time without memory leak time penalty
    plt.plot(
        scenario_nums,
        estimated_times,
        label="Estimated",
        c="r",
        marker="o",
        alpha=0.5,
        linestyle="--",
    )  # estimate of solve time without data transfer overheads

    # use log-log scale
    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Number of Scenarios")
    plt.ylabel("Solve Time (s)")
    plt.legend(title="Algorithm")

    plt.savefig(os.path.join("plots", "solve_times.png"), dpi=300)
    plt.show()

    # plot solution accuracy
    # ==========================
    error_data = {
        "errors": np.concatenate(
            [
                np.array(obj_diffs) * 100,
                np.array(wind_diffs) * 100,
                np.array(storage_diffs) * 100,
            ]
        ),
        "scenarios": scenario_nums * 3,
        "var": ["Objective"] * len(scenario_nums)
        + ["Wind Capacity"] * len(scenario_nums)
        + ["Storage Capacity"] * len(scenario_nums),
    }

    fig, ax = plt.subplots()
    g = sns.barplot(
        data=error_data,
        x="scenarios",
        y="errors",
        hue="var",
        ax=ax,
    )
    plt.xlabel("Number of Scenarios")
    plt.ylabel("Percentage Error (%)")
    ax.legend(title="Variable")

    plt.savefig(os.path.join("plots", "solution_accuracy.png"), dpi=300)
    plt.show()
