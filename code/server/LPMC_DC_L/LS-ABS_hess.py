import sys
sys.path.append("../..")

import os
import time
import json

from algos import OptAlg

from models import LPMC_DC

data_folder = '../../../data/'

if __name__ == "__main__":

    if not os.path.exists('./results'):
        os.makedirs('./results')

    print("Train LPMC_DC_L with LS-ABS and hess")

    model = LPMC_DC(data_folder, file='12_13_14.csv')

    ioa = OptAlg(alg_type='LS-ABS', direction='hess')

    res = {'time': [], 'LL': [], 'epochs': []}

    for i in range(20):

        tmp = model.optimize(ioa, **{'verbose': False, 'max_epochs': 1000, 'batch': 1000})

        res['time'].append(tmp['opti_time'])
        res['LL'].append(tmp['fun'])
        res['epochs'].append(tmp['nep'])

        with open('results/LS-ABS_hess.json', 'w') as outfile:
            json.dump(res, outfile)

        print("{}/20 done!".format(i+1))
