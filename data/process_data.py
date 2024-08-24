"""Process input data into consistent format."""

import os
import numpy as np
import pandas as pd

from configs.config import dataset_dir, wind_years, solar_years, price_years, carbon_years


if __name__ == '__main__':

    hpy = 8760 # hours per year
    raw_data_dir = os.path.join('data','raw')

    # Wind generation data
    dir = os.path.join(raw_data_dir,'wind')
    fpath = os.path.join(dir,'{year}.csv')
    for year in wind_years:
        raw_data = pd.read_csv(fpath.format(year=year),usecols=['electricity'])
        raw_data.rename(columns={'electricity': 'Wind generation [kW/kWp]'}, inplace=True)
        raw_data.head(hpy).to_csv(os.path.join(dataset_dir,'wind',f'{year}.csv'), index=False)

    # Solar generation data
    dir = os.path.join(raw_data_dir,'solar')
    fpath = os.path.join(dir,'Timeseries_51.950_4.100_SA2_1kWp_crystSi_14_40deg_3deg_2010_2019.csv')
    all_data = pd.read_csv(fpath,usecols=['time','P'],skiprows=10,skipfooter=13,engine='python')
    all_data['time'] = pd.to_datetime(all_data['time'],format='%Y%m%d:%H%M')

    for year in solar_years:
        raw_data = all_data.loc[all_data['time'].dt.year == year].copy()
        raw_data.drop(columns='time',inplace=True)
        raw_data = raw_data.apply(lambda x: np.round(x/1000,3))
        raw_data.rename(columns={'P': 'Solar generation [kW/kWp]'}, inplace=True)
        raw_data.head(hpy).to_csv(os.path.join(dataset_dir,'solar',f'{year}.csv'), index=False)

    # Price data
    dir = os.path.join(raw_data_dir,'price')
    fpath = os.path.join(dir,'Day-ahead Prices_{year}.csv')
    for year in price_years:
        raw_data = pd.read_csv(fpath.format(year=year),usecols=['Day-ahead Price [EUR/MWh]'])
        # scale to €300/MWh average price
        raw_data = raw_data.apply(lambda x: x/raw_data['Day-ahead Price [EUR/MWh]'].mean()*300)
        raw_data = raw_data.apply(lambda x: np.round(x/1000,3)) # convert to €/kWh
        raw_data.rename(columns={'Day-ahead Price [EUR/MWh]': 'Electricity price [EUR/kWh]'}, inplace=True)
        raw_data.head(hpy).to_csv(os.path.join(dataset_dir,'price',f'{year}.csv'), index=False)

    # Carbon data
    dir = os.path.join(raw_data_dir,'carbon')
    fpath = os.path.join(dir,'NL_{year}_hourly.csv')
    for year in carbon_years:
        raw_data = pd.read_csv(fpath.format(year=year),usecols=[4],encoding='utf-8')
        raw_data = raw_data.apply(lambda x: np.round(x/1000,3)) # convert to kgCO2e/kWh
        raw_data.rename(columns={raw_data.columns[0]: 'Carbon intensity [kgCO2/kWh]'}, inplace=True)
        raw_data.head(hpy).to_csv(os.path.join(dataset_dir,'carbon',f'{year}.csv'), index=False)