"""Check result files for posterior experiments have been generated correctly."""

import os
import sys
import yaml
import itertools

from configs import get_experiment_config


if __name__ == "__main__":

    expt_id = int(sys.argv[1])
    settings, base_params = get_experiment_config(expt_id)

    available_technologies = list(settings['probability_settings']['storage'].keys())
    tech_combos = ['-'.join(techs) for techs in itertools.combinations(available_technologies, settings['model_settings']['N_technologies'])]

    dir_to_check = os.path.join(*settings['results_dir'],'posterior')

    for z in range(settings['probability_settings']['n_prior_samples']):
        z_dir = os.path.join(dir_to_check,f'z_scenario_{z}')

        if not os.path.exists(z_dir):
            print(f"Missing directory for scenario {z}")
            continue

        for name in ['open','restricted']:
            if not os.path.exists(os.path.join(z_dir,f'{name}_design.yaml')):
                print(f"Missing {name} design file for scenario {z}")
                continue

        for techs in tech_combos:
            if not os.path.exists(os.path.join(z_dir,'design_options',f'{techs}_design.yaml')):
                print(f"Missing {techs} design file for scenario {z}")
            else:
                with open(os.path.join(z_dir,'design_options',f'{techs}_design.yaml'),'r') as f:
                    design = yaml.safe_load(f)
                term_cond = design['solve_stats']['termination_condition']
                if term_cond != 'optimal':
                    print(f"Scenario {z}, {techs} design, solver terminated with status {term_cond}")