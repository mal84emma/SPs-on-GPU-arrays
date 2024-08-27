"""Implementation of Stochastic Programming model of energy system.
**Adapted from Building Design VoI implementation**"""

import numpy as np
import pandas as pd
import xarray as xr
from linopy import Model
from utils.data_handling import ScenarioData

from typing import Iterable


class EnergyModel():

    def __init__(self):

        self.delta_t = 1 # time step in hours

    def generate_SP(
            self,
            scenarios: Iterable[ScenarioData],
            settings: dict
    ):
        """ToDo"""

        # TODO: data validation
        # check all keys present in settings dict
        self.T = settings['T'] # number of time steps
        self.initial_SoC = settings['initial_SoC'] # initial (fractional) state of charge

        assert all([scenario.storage_technologies == scenarios[0].storage_technologies for scenario in scenarios]), "Storage technologies must be consistent across scenarios."
        self.techs = scenarios[0].storage_technologies
        self.max_storage_cap = settings['max_storage_cap'] # maximum storage capacity
        self.N_technologies = settings['N_technologies'] # number of technologies to select

        self.grid_capacity = settings['grid_capacity'] # maximum grid capacity (kW)
        self.solar_capacity_limit = settings['solar_capacity_limit'] # maximum solar capacity (kWp)
        self.capex_budget = settings['capex_budget'] # maximum capital expenditure (€, annualised)

        self.scenarios = scenarios
        self.M = len(scenarios) # number of scenarios

        # set scenario probabilities
        if all([scenario.probability is not None for scenario in scenarios]):
            scenario_weightings = np.array([scenario.probability for scenario in scenarios])
            assert np.isclose(np.sum(scenario_weightings), 1.0), "Scenario weightings must sum to 1."
        else: # assume scenarios equally probable
            scenario_weightings = np.ones(self.M)/self.M


        ## Construct model
        ## ===============
        self.model = Model(force_dim_names=True)

        # Capacity variables
        wind_capacity = self.model.add_variables(lower=0, name='wind_capacity')
        solar_capacity = self.model.add_variables(lower=0, name='solar_capacity')
        storage_capacities = [self.model.add_variables(lower=0, name=f'{tech}_capacity') for tech in self.techs]
        technology_selection = self.model.add_variables(name='tech_selection', binary=True, coords=[pd.RangeIndex(len(self.techs),name='technologies')])

        self.model.add_constraints(solar_capacity, '<=', self.solar_capacity_limit, name='solar_capacity_limit')

        ## Storage technology selection (constraints)
        for i,tech in enumerate(self.techs):
            self.model.add_constraints(storage_capacities[i], '<=', (technology_selection[i]*self.max_storage_cap).to_linexpr(), name=f'{tech}_selection_slack_capacity')
        self.model.add_constraints(technology_selection.sum(), '=', self.N_technologies, name='tech_selection_sum')

        self.scen_obj_contrs = {}
        scen_objectives = []

        ## Dynamics
        for m,scenario  in enumerate(scenarios):

            load = xr.DataArray(scenario.load, coords=[pd.RangeIndex(self.T,name='time')])
            wind = xr.DataArray(scenario.norm_wind_gen[:self.T], coords=[pd.RangeIndex(self.T,name='time')]) * wind_capacity
            solar = xr.DataArray(scenario.norm_solar_gen[:self.T], coords=[pd.RangeIndex(self.T,name='time')]) * solar_capacity
            elec_prices = xr.DataArray(scenario.elec_prices[:self.T].clip(0), coords=[pd.RangeIndex(self.T,name='time')])
            carbon_intensity = xr.DataArray(scenario.carbon_intensity[:self.T], coords=[pd.RangeIndex(self.T,name='time')])

            grid_energy = 0 # net energy flow *from* grid

            for i,tech in enumerate(self.techs):
                stor_cap = storage_capacities[i]
                P_max = scenario.discharge_ratios[i]*stor_cap
                eta = scenario.storage_efficiencies[i]

                # Dynamics decision variables
                SOC = self.model.add_variables(lower=0, name=f'SOC_i{i}_s{m}', coords=[pd.RangeIndex(self.T,name='time')])
                Ein = self.model.add_variables(lower=0, name=f'Ein_i{i}_s{m}', coords=[pd.RangeIndex(self.T,name='time')])
                Eout = self.model.add_variables(lower=0, name=f'Eout_i{i}_s{m}', coords=[pd.RangeIndex(self.T,name='time')])

                # Dynamics constraints
                self.model.add_constraints(self.initial_SoC*stor_cap[0] + -1*SOC[0] + np.sqrt(eta)*Ein[0] - 1/np.sqrt(eta)*Eout[0], '=', 0, name=f'SOC_init_i{i}_s{m}')
                self.model.add_constraints(SOC[:-1] - SOC[1:] + np.sqrt(eta)*Ein[1:] - 1/np.sqrt(eta)*Eout[1:], '=', 0, name=f'SOC_series_i{i}_s{m}')

                self.model.add_constraints(Ein, '<=', P_max*self.delta_t, name=f'Pin_max_i{i}_s{m}')
                self.model.add_constraints(Ein, '<=', P_max*self.delta_t, name=f'Pout_max_i{i}_s{m}')

                self.model.add_constraints(SOC, '<=', stor_cap, name=f'SOC_max_i{i}_s{m}')

                grid_energy += (Ein - Eout)


            wind_curtailment = self.model.add_variables(lower=0, name=f'wind_curtailment_s{m}', coords=[pd.RangeIndex(self.T,name='time')])
            self.model.add_constraints(wind_curtailment, '<=', wind, name=f'wind_curtailment_s{m}')

            grid_energy += -1*(wind - wind_curtailment) + -1*solar + load

            # TODO - fix this!!!
            pos_grid_energy = self.model.add_variables(lower=0, name=f'pos_grid_energy_s{m}', coords=[pd.RangeIndex(self.T,name='time')])
            self.model.add_constraints(pos_grid_energy, '>=', grid_energy, name=f'pos_grid_energy_s{m}')
            # # experimental
            neg_grid_energy = self.model.add_variables(lower=0, name=f'neg_grid_energy_s{m}', coords=[pd.RangeIndex(self.T,name='time')])
            self.model.add_constraints(neg_grid_energy, '<=', -1*grid_energy, name=f'neg_grid_energy_s{m}')
            self.model.add_constraints(grid_energy, '<=', self.grid_capacity*self.delta_t, name=f'pos_grid_limit_s{m}')
            self.model.add_constraints(grid_energy, '>=', -1*self.grid_capacity*self.delta_t, name=f'neg_grid_limit_s{m}')


            ## Scenario objective
            storage_cost = 0
            for i,tech in enumerate(self.techs):
                storage_cost += (scenario.storage_costs[i]/scenario.storage_lifetimes[i])*storage_capacities[i]
            self.scen_obj_contrs[m] = {
                'wind': (scenario.wind_capex/scenario.wind_lifetime + scenario.wind_opex) * wind_capacity,
                'solar': (scenario.solar_capex/scenario.solar_lifetime + scenario.solar_opex) * solar_capacity,
                'storage': storage_cost,
                'elec': pos_grid_energy @ elec_prices - neg_grid_energy @ elec_prices,
                'carbon': pos_grid_energy @ carbon_intensity * scenario.carbon_price
            }

            scen_obj = sum(self.scen_obj_contrs[m].values())
            scen_objectives.append(scen_obj)

            self.model.add_constraints(sum([self.scen_obj_contrs[0][key] for key in ['wind','solar','storage']]), '<=', self.capex_budget, name=f'capex_budget_s{m}')

        print(scenario_weightings @ scen_objectives)
        self.model.add_objective(scenario_weightings @ scen_objectives)