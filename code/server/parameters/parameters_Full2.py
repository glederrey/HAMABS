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

    base_param = {'perc_hybrid': 30, 'thresh_upd': 1, 'count_upd': 2, 'window': 10, 'factor_upd': 2}

    main_params = {'verbose': False, 'nbr_epochs': 1000, 'batch': 1000}
    main_params.update(base_param)

    draws = 20

    res = {}

    tmp_res = {'time': [], 'LL': [], 'epochs': []}

    print("Start with factor_upd")

    param_fu = [1.1, 1.2, 1.3, 1.4, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10]

    res['factor_upd'] = {}

    for fu in param_fu:

        tmp_res = {'time': [], 'LL': [], 'epochs': []}

        print("  Value: {}".format(fu))

        for i in range(draws):

            main_params['factor_upd'] = fu

            tmp = model.optimize(ioa, **main_params)

            tmp_res['time'].append(tmp['opti_time'])
            tmp_res['LL'].append(tmp['fun'])
            tmp_res['epochs'].append(tmp['nep'])

            res['factor_upd'][fu] = tmp_res

            with open('results/parameters_Full2.json', 'w') as outfile:
                json.dump(res, outfile)

            print("{}/{} done!".format(i + 1, draws))
        print("")

    main_params['factor_upd'] = 2

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

            with open('results/parameters_Full2.json', 'w') as outfile:
                json.dump(res, outfile)

            print("{}/{} done!".format(i + 1, draws))
        print("")

    main_params['perc_hybrid'] = 30

    print("Start with stopping crit")

    param_sc = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]

    res['stop_crit'] = {}

    for sc in param_sc:

        tmp_res = {'time': [], 'LL': [], 'epochs': []}

        print("  Value: {}".format(sc))

        for i in range(draws):

            main_params['thresh'] = sc

            tmp = model.optimize(ioa, **main_params)

            tmp_res['time'].append(tmp['opti_time'])
            tmp_res['LL'].append(tmp['fun'])
            tmp_res['epochs'].append(tmp['nep'])

            res['stop_crit'][sc] = tmp_res

            with open('results/parameters_Full2.json', 'w') as outfile:
                json.dump(res, outfile)

            print("{}/{} done!".format(i + 1, draws))
        print("")

    print("DONE!")
