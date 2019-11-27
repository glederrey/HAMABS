import sys
sys.path.append("../..")

import os
import time
import json

from algos import OptAlg

from models import LPMC_DC

data_folder = '../../../data/'

if __name__ == "__main__":

    parameters = {''}

    if not os.path.exists('./results'):
        os.makedirs('./results')

    print("Testing parameters for LPMC_RR_L")

    model = LPMC_DC(data_folder, file='12_13_14.csv')

    ioa = OptAlg(alg_type='LS-ABS', direction='hybrid-inv')

    base_param = {'perc_hybrid': 30, 'thresh_upd': 1, 'count_upd': 2, 'window': 10, 'factor_upd': 2, 'stop_crit': 1e-6}

    main_params = {'verbose': True, 'nbr_epochs': 1000, 'batch': 1000}
    main_params.update(base_param)

    draws = 10

    res = {}

    print("Start with perc_hybrid")

    param_ph = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

    res['perc_hybrid'] = {}

    for ph in param_ph:

        tmp_res = {'time': [], 'LL': [], 'epochs': []}

        print("  Value: {}".format(ph))

        for i in range(draws):
            main_params['perc_hybrid'] = ph

            tmp = model.optimize(ioa, **main_params)

            tmp_res['time'].append(tmp['opti_time'])
            tmp_res['LL'].append(tmp['fun'])
            tmp_res['epochs'].append(tmp['nep'])

            res['perc_hybrid'][ph] = tmp_res

            with open('results/DC_hybrid.json', 'w') as outfile:
                json.dump(res, outfile)

            print("{}/{} done!".format(i + 1, draws))
        print("")
