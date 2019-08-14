import sys
sys.path.append("../..")

import os
import time
import json

from models import LPMC_MNL_DC

data_folder = '../../../data/'

if __name__ == "__main__":

    if not os.path.exists('./results'):
        os.makedirs('./results')

    print("Train LPMC_MNL_DC with current Biogeme optimization algorithm.")

    model = LPMC_MNL_DC(data_folder, file='12.csv')

    res = {'time': [], 'LL': [], 'epochs': []}

    for i in range(20):

        start = time.time()
        results = model.biogeme.estimate(gtol=0.020730678636620467)
        tme = time.time() - start

        res['time'].append(tme)
        res['LL'].append(results.data.logLike)
        res['epochs'].append(model.biogeme.numberOfIterations)

        with open('results/biogeme_bfgs_tol.json', 'w') as outfile:
            json.dump(res, outfile)

        # Delete pickle files
        all_files = os.listdir()

        for item in all_files:
            if item.endswith(".pickle"):
                os.remove(item)

        print("{}/20 done!".format(i+1))
