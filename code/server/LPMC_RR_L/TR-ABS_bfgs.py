import sys
sys.path.append("../..")

import os
import time
import json

from algos import OptAlg

from models import LPMC_RR

data_folder = '../../../data/'

if __name__ == "__main__":

    if not os.path.exists('./results'):
        os.makedirs('./results')

    print("Train LPMC_RR_L with TR-ABS and bfgs")

    model = LPMC_RR(data_folder, file='12_13_14.csv')

    ioa = OptAlg(alg_type='TR-ABS', direction='bfgs')

    res = {'time': [], 'LL': [], 'epochs': []}

    for i in range(20):

        tmp = model.optimize(ioa, **{'verbose': False, 'max_epochs': 1000, 'batch': 1000})

        res['time'].append(tmp['opti_time'])
        res['LL'].append(tmp['fun'])
        res['epochs'].append(tmp['nep'])

        with open('results/TR-ABS_bfgs.json', 'w') as outfile:
            json.dump(res, outfile)

        print("{}/20 done!".format(i+1))
