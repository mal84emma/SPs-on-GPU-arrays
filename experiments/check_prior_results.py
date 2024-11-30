"""Check result files for prior experiments have been generated correctly,
and create best_design file."""

import os
import sys
import yaml
import shutil
import itertools

from configs import get_experiment_config


if __name__ == "__main__":

    expt_name = str(sys.argv[1])
    settings, base_params = get_experiment_config(expt_name)

    available_technologies = list(settings['probability_settings']['storage'].keys())
    tech_combos = ['-'.join(techs) for techs in itertools.combinations(available_technologies, settings['model_settings']['N_technologies'])]

    dir_to_check = os.path.join(*settings['results_dir'],'prior')


    best_objective = float('inf')
    best_techs = None
    results_incomplete = False

    for techs in tech_combos:
        if not os.path.exists(os.path.join(dir_to_check,f'{techs}_design.yaml')):
            print(f"Missing {techs} design file")
            results_incomplete = True
        else:
            with open(os.path.join(dir_to_check,f'{techs}_design.yaml'),'r') as f:
                design = yaml.safe_load(f)
                obj = design['overall_objective']['overall_objective']

                if obj < best_objective:
                    best_objective = obj
                    best_techs = techs

    if not results_incomplete:
        print(f"Best design is {best_techs}")
        shutil.copy(os.path.join(dir_to_check,f'{best_techs}_design.yaml'),os.path.join(dir_to_check,f'best_design.yaml'))