import sys
sys.path.append("../..")

import os
import time
import json

from algos import OptAlg

from models import LPMC_Full

data_folder = '../../../data/'

if __name__ == "__main__":

    if not os.path.exists('./results'):
        os.makedirs('./results')

    print("Train LPMC_Full_big with LS-ABS and hybrid-inv")

    model = LPMC_Full(data_folder, file='12_13_14.csv')

    ioa = OptAlg(alg_type='LS-ABS', direction='hybrid-inv')

    res = model.optimize(ioa, **{'verbose': False, 'nbr_epochs': 1000, 'batch': 1000})

    with open('results/res.json', 'w') as outfile:
        json.dump(res, outfile)
