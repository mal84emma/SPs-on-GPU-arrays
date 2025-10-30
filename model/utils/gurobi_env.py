"""Gurobi helper functions."""

import os
import gurobipy as gp


def get_Gurobi_WLS_env(silence=False):
    """Create Gurobi environment using WLS license."""

    try:
        # Read license file
        license_path = os.path.join('resources','gurobi.lic')
        with open(license_path, 'r') as file:
            license_lines = [line.rstrip() for line in file]

        required_details = ['WLSACCESSID','LICENSEID','WLSSECRET']

        # Create Gurobi environment
        e = gp.Env(empty=True)
        for detail in required_details:
            if silence:
                e.setParam('OutputFlag',0)
            key, = [line.split('=')[1] for line in license_lines if detail in line]
            if detail == 'LICENSEID':
                key = int(key)
            e.setParam(detail,key)

        # Activate environment
        e.start()

        return e

    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == '__main__':

    env = get_Gurobi_WLS_env()