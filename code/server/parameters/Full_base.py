import sys
sys.path.append("../..")

import os
import time
import json

from algos import OptAlg

from models import LPMC_Full

data_folder = '../../../data/'

if __name__ == "__main__":

    parameters = {''}

    if not os.path.exists('./results'):
        os.makedirs('./results')

    print("Testing parameters for LPMC_Full_L")

    model = LPMC_Full(data_folder, file='12_13_14.csv')

    ioa = OptAlg(alg_type='LS-ABS', direction='hybrid-inv')

    base_param = {'perc_hybrid': 30, 'thresh_upd': 1, 'count_upd': 2, 'window': 10, 'factor_upd': 2, 'stop_crit': 1e-6}

    main_params = {'verbose': False, 'nbr_epochs': 1000, 'batch': 1000}
    main_params.update(base_param)

    draws = 20

    res = {}

    tmp_res = {'time': [], 'LL': [], 'epochs': []}

    print("Start with base params")

    for i in range(draws):

        tmp = model.optimize(ioa, **main_params)

        tmp_res['time'].append(tmp['opti_time'])
        tmp_res['LL'].append(tmp['fun'])
        tmp_res['epochs'].append(tmp['nep'])

        res['base'] = tmp_res

        with open('results/Full_base.json', 'w') as outfile:
            json.dump(res, outfile)

        print("{}/{} done!".format(i+1, draws))