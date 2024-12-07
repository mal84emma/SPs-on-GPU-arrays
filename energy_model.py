"""Implementation of Stochastic Programming model of energy system.
**Adapted from Building Design VoI implementation**"""

import os
import yaml
import time
import numpy as np
import pandas as pd
import xarray as xr
from linopy import Model
from utils.data_handling import ScenarioData

from typing import Iterable, Tuple, Dict


class EnergyModel():

    def __init__(self):

        self.delta_t = 1 # time step in hours

    def generate_SP(
            self,
            scenarios: Iterable[ScenarioData],
            settings: Dict,
            set_design: Dict = None
    ) -> Model:
        """Construct LP (scenario program) model of energy park, using
        specified list of scenarios and setting dictionary.

        Args:
            scenarios (Iterable[ScenarioData]): List of ScenarioData objects
                defining scenarios to include in SP.
            settings (Dict): Dictionary of model settings. (`model_settings`
                from `settings.yaml`)
            set_design (Dict, optional): Dictionary specifying system design to
                run optimization in operational mode. Defaults to None.

        Returns:
            (Model): LP model object with set constraints and objective.
        """

        ## Setup: data validation & formatting
        ## ===================================
        expected_keys = ['T','initial_SoC','N_technologies','allow_elec_purchase','grid_capacity','solar_capacity_limit','capex_budget']
        assert all([key in settings for key in expected_keys]), "Settings dict must contain all required keys."

        assert type(settings['T']) == int, "T must be an integer."
        assert settings['T'] > 0, "T must be a positive integer."
        self.T = settings['T'] # number of time steps

        assert type(settings['initial_SoC']) == float, "initial_SoC must be a float."
        assert 0 <= settings['initial_SoC'] <= 1, "initial_SoC must be between 0 and 1."
        self.initial_SoC = settings['initial_SoC'] # initial (fractional) state of charge

        assert 'storage_technologies' in settings.keys(), "`storage_technologies` must be specified in settings (list of technologies used in specified system)."
        self.techs = settings['storage_technologies'] # storage technologies
        assert len(self.techs) > 0, "At least one storage technology must be used."
        if self.techs == ['none']:
            self.techs = []
        else:
            assert len(self.techs) == settings['N_technologies'], "Number of used storage technologies must match `N_technologies`."
            assert all([tech in scenario.storage_technologies for tech in self.techs for scenario in scenarios]), "Storage technologies must be available in all scenarios."

        assert type(settings['allow_elec_purchase']) == bool, "allow_elec_purchase must be a boolean."
        self.allow_elec_purchase = settings['allow_elec_purchase'] # allow electricity purchase from grid
        self.grid_capacity = settings['grid_capacity'] # maximum grid capacity (kW)
        self.solar_capacity_limit = settings['solar_capacity_limit'] # maximum solar capacity (kWp)
        self.capex_budget = settings['capex_budget'] # maximum capital expenditure (€, annualised)

        assert type(settings['use_CVaR']) == bool, "use_CVaR must be a boolean."
        self.use_CVaR = settings['use_CVaR'] # use Conditional Value at Risk in objective
        if self.use_CVaR:
            assert 0 < settings['CVaR_alpha'] < 0.5, "CVaR_alpha must be between 0 and 0.5."
            assert settings['CVaR_beta'] > 0, "CVaR_beta must be positive."
            self.alpha = settings['CVaR_alpha'] # confidence level
            self.beta = settings['CVaR_beta'] # risk aversion parameter

        if set_design is not None:
            assert all([key in set_design for key in ['wind_capacity','solar_capacity']]), "Design dict must contain keys 'wind_capacity' and 'solar_capacity'."
            assert set_design['storage_technologies'] == self.techs, "Storage technologies in set_design must match those in settings."

        self.scenarios = scenarios
        self.M = len(scenarios) # number of scenarios

        # set scenario probabilities
        if all([scenario.probability is not None for scenario in scenarios]):
            self.scenario_weightings = np.array([scenario.probability for scenario in scenarios])
            assert np.isclose(np.sum(self.scenario_weightings), 1.0, rtol=1e-3),\
                f"Scenario weightings must sum to 1.  Currently sum to {np.sum(self.scenario_weightings)}"
        else: # assume scenarios equally probable
            self.scenario_weightings = np.ones(self.M)/self.M


        ## Construct model
        ## ===============
        self.model: Model = Model(force_dim_names=True)

        ## Capacity variables
        wind_capacity = self.model.add_variables(lower=0, name='wind_capacity')
        solar_capacity = self.model.add_variables(lower=0, name='solar_capacity')
        storage_capacities = {tech: self.model.add_variables(lower=0, name=f'{tech}_capacity') for tech in self.techs}

        if set_design is None:
            self.model.add_constraints(solar_capacity, '<=', self.solar_capacity_limit, name='solar_capacity_limit')
        else: # set capacity variables to specified design values
            self.model.add_constraints(solar_capacity, '=', set_design['solar_capacity']['value'], name='solar_capacity')
            self.model.add_constraints(wind_capacity, '=', set_design['wind_capacity']['value'], name='wind_capacity')
            for tech in self.techs:
                self.model.add_constraints(storage_capacities[tech], '=', set_design['storage_capacities'][tech]['value'], name=f'{tech}_capacity')

        # access objects
        self.scen_obj_contrs = {}
        self.grid_energies = {}
        self.scenario_objectives = []

        ## Scenarios
        for m,scenario  in enumerate(scenarios):

            load = xr.DataArray(scenario.load[:self.T], coords=[pd.RangeIndex(self.T,name='time')])
            wind = xr.DataArray(scenario.norm_wind_gen[:self.T], coords=[pd.RangeIndex(self.T,name='time')]) * wind_capacity
            solar = xr.DataArray(scenario.norm_solar_gen[:self.T], coords=[pd.RangeIndex(self.T,name='time')]) * solar_capacity
            elec_prices = xr.DataArray(scenario.elec_prices[:self.T], coords=[pd.RangeIndex(self.T,name='time')]) # .clip(0)
            carbon_intensity = xr.DataArray(scenario.carbon_intensity[:self.T], coords=[pd.RangeIndex(self.T,name='time')])

            ## Dynamics
            battery_energy = 0 # net energy flow *into* batteries

            for tech in self.techs:
                storage_capacity = storage_capacities[tech]
                P_max = scenario.discharge_ratios[tech]*storage_capacity
                E_min = (1-scenario.depths_of_discharge[tech])*storage_capacity
                eta = scenario.storage_efficiencies[tech]

                # Dynamics decision variables
                SOC = self.model.add_variables(lower=0, name=f'SOC_{tech}_s{m}', coords=[pd.RangeIndex(self.T,name='time')])
                Ein = self.model.add_variables(lower=0, name=f'Ein_{tech}_s{m}', coords=[pd.RangeIndex(self.T,name='time')])
                Eout = self.model.add_variables(lower=0, name=f'Eout_{tech}_s{m}', coords=[pd.RangeIndex(self.T,name='time')])

                # Dynamics constraints
                self.model.add_constraints(self.initial_SoC*storage_capacity[0] + -1*SOC[0] + np.sqrt(eta)*Ein[0] - 1/np.sqrt(eta)*Eout[0], '=', 0, name=f'SOC_init_{tech}_s{m}')
                self.model.add_constraints(SOC[:-1] - SOC[1:] + np.sqrt(eta)*Ein[1:] - 1/np.sqrt(eta)*Eout[1:], '=', 0, name=f'SOC_series_{tech}_s{m}')

                self.model.add_constraints(Ein, '<=', P_max*self.delta_t, name=f'Pin_max_{tech}_s{m}')
                self.model.add_constraints(Eout, '<=', P_max*self.delta_t, name=f'Pout_max_{tech}_s{m}')

                self.model.add_constraints(SOC, '<=', storage_capacity, name=f'SOC_max_{tech}_s{m}')
                self.model.add_constraints(SOC, '>=', E_min, name=f'SOC_min_{tech}_s{m}')

                battery_energy += (Ein - Eout)


            generation_curtailment = self.model.add_variables(lower=0, name=f'generation_curtailment_s{m}', coords=[pd.RangeIndex(self.T,name='time')])
            self.model.add_constraints(generation_curtailment, '<=', wind + solar, name=f'generation_curtailment_s{m}')

            supplied_energy = -1*(wind + solar - generation_curtailment) + battery_energy
            # still consumption +ve, weird eqn order due to xarray madness
            grid_energy = supplied_energy + load
            self.grid_energies[m] = grid_energy

            if self.allow_elec_purchase: # if grid import allowed
                self.model.add_constraints(grid_energy, '<=', self.grid_capacity*self.delta_t, name=f'pos_grid_limit_s{m}')
            else:
                self.model.add_constraints(grid_energy, '<=', 0, name=f'green_power_only_s{m}')
            self.model.add_constraints(grid_energy, '>=', -1*self.grid_capacity*self.delta_t, name=f'neg_grid_limit_s{m}')

            pos_grid_energy = self.model.add_variables(lower=0, name=f'pos_grid_energy_s{m}', coords=[pd.RangeIndex(self.T,name='time')]) # slack variable for carbon emissions
            self.model.add_constraints(pos_grid_energy, '>=', grid_energy, name=f'pos_grid_energy_s{m}')


            ## Scenario objective
            storage_cost = 0
            for tech in self.techs:
                storage_cost += (scenario.storage_costs[tech]/scenario.storage_lifetimes[tech])*storage_capacities[tech]
            self.scen_obj_contrs[m] = {
                'wind': (scenario.wind_capex/scenario.wind_lifetime + scenario.wind_opex) * wind_capacity,
                'solar': (scenario.solar_capex/scenario.solar_lifetime + scenario.solar_opex) * solar_capacity,
                'storage': storage_cost,
                'elec': supplied_energy @ elec_prices, # electricity cost without plant usage (constants not allowed in objective)
                'carbon': pos_grid_energy @ carbon_intensity * scenario.carbon_price
            }

            self.scenario_objectives.append(sum(self.scen_obj_contrs[m].values()))
            self.model.add_constraints(sum([self.scen_obj_contrs[m][key] for key in ['wind','solar','storage']]), '<=', self.capex_budget, name=f'capex_budget_s{m}')
            # planned capacities must be within budget in all scenarios - capacity decision made before costs perfectly known

        ## Overall objective
        self.scenario_objectives = np.array(self.scenario_objectives)
        objective = self.scenario_weightings @ self.scenario_objectives

        if self.use_CVaR: # add CVaR objective contribution & constraints
            xi = self.model.add_variables(name='CVaR_value_threshold')
            etas = self.model.add_variables(lower=0, name='CVaR_slack', coords=[pd.RangeIndex(self.M,name='scenarios')])

            for m in range(self.M): # add eta constraints per scenario (due to xarray datatype headaches)
                self.model.add_constraints(etas[m] + xi[0], '>=', self.scenario_objectives[m], name=f'CVaR_threshold_s{m}')

            self.CVaR_obj_contribution = self.beta*(xi + 1/self.alpha*(self.scenario_weightings*etas).sum())
            objective += self.CVaR_obj_contribution
        # endif

        self.model.add_objective(objective, sense='min')

        return self.model


    def solve(self, **kwargs):
        """Solve constructed model and report corrected objective value."""

        start_time = time.time()

        # ToDo arg parsing for solvers
        self.model.solve(**kwargs)

        self.load_elec_cost = self.scenario_weightings @ [self.scenarios[m].load[:self.T] @ self.scenarios[m].elec_prices[:self.T] for m in range(self.M)]
        self.corrected_objective = self.model.objective.value + self.load_elec_cost

        self.run_time = time.time() - start_time

        return self.corrected_objective


    def get_flared_energy(self):
        """ToDo ... grid constraints -> energy dumping is economic
        need for +ve & -ve storage flow means model allows this
        this is also done in PyPSA https://github.com/PyPSA/PyPSA/blob/master/test/test_lopf_basic_constraints.py#L22"""

        self.energy_flares = {f's{m}': {} for m in range(self.M)}

        for m in range(self.M):
            total_dumped = 0

            for tech in self.techs:
                eta = self.scenarios[m].storage_efficiencies[tech]
                e2 = getattr(self.model.variables,f'Ein_{tech}_s{m}').solution * getattr(self.model.variables,f'Eout_{tech}_s{m}').solution

                Ein_dumps = getattr(self.model.variables,f'Ein_{tech}_s{m}').solution.where(e2 > 0, 0)
                Eout_dumps = getattr(self.model.variables,f'Eout_{tech}_s{m}').solution.where(e2 > 0, 0)
                net_energy_in = Ein_dumps - Eout_dumps
                net_energy_gain = np.sqrt(eta)*Ein_dumps - 1/np.sqrt(eta)*Eout_dumps
                dumped_energy = net_energy_in - net_energy_gain

                self.energy_flares[f's{m}'][tech] = dumped_energy
                total_dumped += dumped_energy

            self.energy_flares[f's{m}'].update({
                'total_energy_dump': total_dumped,
                'generation_curtailment': self.model.variables[f'generation_curtailment_s{m}'].solution
            })

        return self.energy_flares


    def get_storage_cycles(self):
        """ToDo ... check battery cycling"""

        self.storage_cycles = {}

        for m in range(self.M):
            self.storage_cycles[f's{m}'] = {}
            for tech in self.techs:
                if self.model.variables[f'{tech}_capacity'].solution > 0:
                    # ein = getattr(self.model.variables,f'Ein_{tech}_s{m}').solution
                    # eout = getattr(self.model.variables,f'Eout_{tech}_s{m}').solution
                    # total_energy_flow = np.sum(np.abs(ein - eout))
                    SOC = getattr(self.model.variables,f'SOC_{tech}_s{m}').solution
                    energy_deltas = np.diff(SOC)
                    total_energy_flow = np.sum(np.abs(energy_deltas))

                    num_cycles = total_energy_flow / (2*self.model.variables[f'{tech}_capacity'].solution.values)
                    # NOTE: not correcting for depth of discharge here
                    self.storage_cycles[f's{m}'][tech] = num_cycles

        return self.storage_cycles


    def save_results(self, fpath: str) -> Tuple[Dict]:
        """Save optimized design and objective values to yaml file.

        Args:
            fpath (str): Path to save yaml file. If '' then no file is saved.

        Returns:
            Tuple[Dict]: Dictionaries of design, overall objective, and scenario objective contributions.
        """

        solve_stats_dict = {
            'termination_condition': self.model.termination_condition,
            'run_time': {
                'unit': 's',
                'value': self.run_time
            }
            # TODO: can I get more solution stats? e.g. optimality gap, constraint violations, etc.
        }

        design_dict = {
            'wind_capacity': {
                'unit': 'kW',
                'value': float(self.model.variables['wind_capacity'].solution.values),
            },
            'solar_capacity': {
                'unit': 'kWp',
                'value': float(self.model.variables['solar_capacity'].solution.values),
            },
            'storage_technologies': self.techs,
            'storage_capacities': {
                tech: {
                    'unit': 'kWh',
                    'value': float(self.model.variables[f'{tech}_capacity'].solution.values),
                } for tech in self.techs
            }
        }

        overall_storage_cost = float(self.scenario_weightings @ [self.scen_obj_contrs[m]['storage'].solution.values for m in range(self.M)]) if self.techs else 0.0
        overall_objective_dict = {
            'units': 'Euros',
            'overall_objective': float(self.corrected_objective),
            'overall_wind_cost': float(self.scenario_weightings @ [self.scen_obj_contrs[m]['wind'].solution.values for m in range(self.M)]),
            'overall_solar_cost': float(self.scenario_weightings @ [self.scen_obj_contrs[m]['solar'].solution.values for m in range(self.M)]),
            'overall_storage_cost': overall_storage_cost,
            'overall_elec_cost': float(self.scenario_weightings @ [self.scen_obj_contrs[m]['elec'].solution.values for m in range(self.M)]) + float(self.load_elec_cost),
            'overall_carbon_cost': float(self.scenario_weightings @ [self.scen_obj_contrs[m]['carbon'].solution.values for m in range(self.M)])
        }
        if self.use_CVaR:
            overall_objective_dict['overall_CVaR_contr'] = float(self.CVaR_obj_contribution.solution.values)

        scenario_objective_contributions_dict = {'units': 'Euros'}
        for m in range(self.M):
            id = self.scenarios[m].id or m
            scen_storage_cost = float(self.scen_obj_contrs[m]['storage'].solution.values) if self.techs else 0.0
            scen_cost_dict = {
                    'wind_cost': float(self.scen_obj_contrs[m]['wind'].solution.values),
                    'solar_cost': float(self.scen_obj_contrs[m]['solar'].solution.values),
                    'storage_cost': scen_storage_cost,
                    'elec_cost': float(self.scen_obj_contrs[m]['elec'].solution.values) + float(self.scenarios[m].load[:self.T] @ self.scenarios[m].elec_prices[:self.T]),
                    'carbon_cost': float(self.scen_obj_contrs[m]['carbon'].solution.values)
                }
            scen_cost_dict['total_cost'] = sum(scen_cost_dict.values())
            scen_cost_dict['probability'] = float(self.scenario_weightings[m])
            scenario_objective_contributions_dict.update({f'scenario_{id}': scen_cost_dict})


        if os.path.dirname(fpath) != '':
            if not os.path.exists(os.path.dirname(fpath)):
                os.makedirs(os.path.dirname(fpath))

        with open(fpath, 'w') as f:
            yaml.dump({
                'solve_stats': solve_stats_dict,
                'design': design_dict,
                'overall_objective': overall_objective_dict,
                'scenario_objective_contributions': scenario_objective_contributions_dict
            }, f, sort_keys=False)

        return solve_stats_dict, design_dict, overall_objective_dict, scenario_objective_contributions_dict

    def save_scenarios(self, dir: str):
        """Save scenario data to yaml files.

        Args:
            dir (str): Path to directory to save scenario files.
        """
        for m,scenario in enumerate(self.scenarios):
            scenario.to_file(os.path.join(dir,f'scenario_{m}.yaml'))
