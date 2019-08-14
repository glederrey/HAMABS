import sys
sys.path.append("../..")

import os
import time
import json

from algos import OptAlg

from models import LPMC_MNL_RR

data_folder = '../../../data/'

if __name__ == "__main__":

    if not os.path.exists('./results'):
        os.makedirs('./results')

    print("Test different percentages for the LS-ABS hybrid-inv\n")

    model = LPMC_MNL_RR(data_folder, file='12_13_14.csv')

    ioa = OptAlg(alg_type='LS-ABS', direction='hybrid-inv')

    percs = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    res = {'percs': percs}

    for p in percs:

        print("Start with perc of {}".format(p))

        res_tmp = {'time': [], 'LL': [], 'epochs': []}
        for i in range(20):
            start = time.time()
            tmp = model.optimize(ioa, **{'verbose': False, 'nbr_epochs': 1000, 'batch': 0.01, 'perc_hybrid': p})
            tme = time.time() - start

            res_tmp['time'].append(tme)
            res_tmp['LL'].append(tmp['fun'])
            res_tmp['epochs'].append(tmp['nep'])

            print("  {}/20 done!".format(i + 1))

            res[p] = res_tmp

            with open('results/res.json', 'w') as outfile:
                json.dump(res, outfile)

        print("\n")

    print("Start with LS-ABS + bfgs-inv")
    ioa = OptAlg(alg_type='LS-ABS', direction='bfgs-inv')

    res_tmp = {'time': [], 'LL': [], 'epochs': []}
    for i in range(20):
        start = time.time()
        tmp = model.optimize(ioa, **{'verbose': False, 'nbr_epochs': 1000, 'batch': 0.01})
        tme = time.time() - start

        res_tmp['time'].append(tme)
        res_tmp['LL'].append(tmp['fun'])
        res_tmp['epochs'].append(tmp['nep'])

        print("  {}/20 done!".format(i + 1))

        res['bfgs-inv'] = res_tmp

        with open('results/res.json', 'w') as outfile:
            json.dump(res, outfile)

    print("\n")

    print("Start with LS-ABS + hessian")
    ioa = OptAlg(alg_type='LS-ABS', direction='hess')

    res_tmp = {'time': [], 'LL': [], 'epochs': []}
    for i in range(20):
        start = time.time()
        tmp = model.optimize(ioa, **{'verbose': False, 'nbr_epochs': 1000, 'batch': 0.01})
        tme = time.time() - start

        res_tmp['time'].append(tme)
        res_tmp['LL'].append(tmp['fun'])
        res_tmp['epochs'].append(tmp['nep'])

        print("  {}/20 done!".format(i + 1))

        res['hess'] = res_tmp

        with open('results/res.json', 'w') as outfile:
            json.dump(res, outfile)

    print("\n")

