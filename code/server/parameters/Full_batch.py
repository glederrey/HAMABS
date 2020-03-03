import sys

sys.path.append("../..")

import os
import time
import json
import numpy as np

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

    base_param = {'perc_hybrid': 30, 'thresh_upd': 1, 'count_upd': 2, 'window': 10, 'factor_upd': 2}

    main_params = {'verbose': False, 'max_epochs': 1000, 'batch': 1000}
    main_params.update(base_param)

    draws = 20

    res = {}

    print("Start with init batch size")

    param_batch = np.array([1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, main_params['batch'] / len(model.biogeme.database.data)])
    param_batch.sort()

    res['batch'] = {}

    for b in param_batch:

        tmp_res = {'time': [], 'LL': [], 'epochs': []}

        print("  Value: {}".format(b))

        main_params['batch'] = b

        for i in range(draws):

            tmp = model.optimize(ioa, **main_params)

            tmp_res['time'].append(tmp['opti_time'])
            tmp_res['LL'].append(tmp['fun'])
            tmp_res['epochs'].append(tmp['nep'])

            res['batch'][b] = tmp_res

            with open('results/Full_batch.json', 'w') as outfile:
                json.dump(res, outfile)

            print("{}/{} done!".format(i + 1, draws))
        print("")
