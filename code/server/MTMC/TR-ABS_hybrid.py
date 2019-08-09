import sys
sys.path.append("../..")

import os
import time
import json

from algos import OptAlg

from models import MTMC

data_folder = '../../../data/'

if __name__ == "__main__":

    if not os.path.exists('./results'):
        os.makedirs('./results')

    print("Train MTMC with TR-ABS and hybrid")

    model = MTMC(data_folder)

    ioa = OptAlg(alg_type='TR-ABS', direction='hybrid')

    res = {'time': [], 'LL': [], 'epochs': []}

    for i in range(20):

        start = time.time()
        tmp = model.optimize(ioa, **{'verbose': False, 'nbr_epochs': 1000, 'batch': 1000})
        tme = time.time() - start

        res['time'].append(tme)
        res['LL'].append(tmp['fun'])
        res['epochs'].append(tmp['nep'])

        with open('results/TR-ABS_hybrid.json', 'w') as outfile:
            json.dump(res, outfile)

        print("{}/20 done!".format(i+1))