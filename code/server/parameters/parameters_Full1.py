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

    print("Start with base params")

    for i in range(draws):

        tmp = model.optimize(ioa, **main_params)

        tmp_res['time'].append(tmp['opti_time'])
        tmp_res['LL'].append(tmp['fun'])
        tmp_res['epochs'].append(tmp['nep'])

        res['base'] = tmp_res

        with open('results/parameters_Full1.json', 'w') as outfile:
            json.dump(res, outfile)

        print("{}/{} done!".format(i+1, draws))

    print("")
    print("Start with thresh_upd")

    param_tu = [1e-1, 2e-1, 5e-1, 1, 2, 5, 10, 20, 50, 100]

    res['thresh_upd'] = {}

    for tu in param_tu:

        tmp_res = {'time': [], 'LL': [], 'epochs': []}

        print("  Value: {}".format(tu))

        for i in range(draws):

            main_params['thresh_upd'] = tu

            tmp = model.optimize(ioa, **main_params)

            tmp_res['time'].append(tmp['opti_time'])
            tmp_res['LL'].append(tmp['fun'])
            tmp_res['epochs'].append(tmp['nep'])

            res['thresh_upd'][tu] = tmp_res

            with open('results/parameters_Full1.json', 'w') as outfile:
                json.dump(res, outfile)

            print("{}/{} done!".format(i + 1, draws))
        print("")

    main_params['thresh_upd'] = 1

    print("Start with count_upd")

    param_co = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    res['count_upd'] = {}

    for co in param_co:

        tmp_res = {'time': [], 'LL': [], 'epochs': []}

        print("  Value: {}".format(co))

        for i in range(draws):

            main_params['count_upd'] = co

            tmp = model.optimize(ioa, **main_params)

            tmp_res['time'].append(tmp['opti_time'])
            tmp_res['LL'].append(tmp['fun'])
            tmp_res['epochs'].append(tmp['nep'])

            res['count_upd'][co] = tmp_res

            with open('results/parameters_Full1.json', 'w') as outfile:
                json.dump(res, outfile)

            print("{}/{} done!".format(i + 1, draws))
        print("")

    main_params['count_upd'] = 2

    print("Start with window")

    param_win = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    res['window'] = {}

    for win in param_win:

        tmp_res = {'time': [], 'LL': [], 'epochs': []}

        print("  Value: {}".format(win))

        for i in range(draws):

            main_params['window'] = win

            tmp = model.optimize(ioa, **main_params)

            tmp_res['time'].append(tmp['opti_time'])
            tmp_res['LL'].append(tmp['fun'])
            tmp_res['epochs'].append(tmp['nep'])

            res['window'][win] = tmp_res

            with open('results/parameters_Full1.json', 'w') as outfile:
                json.dump(res, outfile)

            print("{}/{} done!".format(i + 1, draws))
        print("")

    main_params['window'] = 10

    print("DONE!")
