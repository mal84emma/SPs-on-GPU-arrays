"""Download wind data from renewables.ninja."""

import pandas as pd
import numpy as np
import os
import json
import time
import requests
import urllib3
import yaml

from configs.config import wind_years, wind_location


if __name__ == '__main__':

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # get API token
    with open(os.path.join('resources','RNapi.yaml')) as f:
        api_creds = yaml.load(f, Loader=yaml.FullLoader)
        token = api_creds['credentials']['token']
        url = 'https://www.renewables.ninja/api/data/wind'

    s = requests.session()
    s.headers = {'Authorization': 'Token ' + token}

    args = { # common API args for all
        'lat': wind_location[0],
        'lon': wind_location[1],
        'dataset': 'merra2',
        'capacity': 1.0,
        'height': 125,
        'turbine': 'Vestas V164 9500',
        'format': 'json',
        'raw': 'true'
    }

    for i,year in enumerate(wind_years):
        args['date_from'] = f'{year}-01-01'
        args['date_to'] = f'{year}-12-31'

        time.sleep(10 if i > 0 else 0) # used to space out api calls
        r = s.get(url, params=args, verify=False)
        if r.status_code != 200:
            print('Error (' + str(r.status_code) + ') getting data for year ' + str(year))
            print(r.text)
            cont = input("Data pull failed. Save partial data set? (y/n): ")
            if (cont not in ['yes','y','Yes','Y','yep']):
                    pull_successful = False
                    raise
            break
        print('Data for ' + str(year) + ' pulled successfully')

        parsed_response = json.loads(r.text)
        metadata = parsed_response['metadata']

        data = pd.read_json(json.dumps(parsed_response['data']), orient='index')
        data.to_csv(os.path.join('data','raw','wind',f'{year}.csv'))