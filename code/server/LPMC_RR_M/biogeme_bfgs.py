import sys
sys.path.append("../..")

import os
import time
import json

from models import LPMC_RR

data_folder = '../../../data/'

if __name__ == "__main__":

    if not os.path.exists('./results'):
        os.makedirs('./results')

    print("Train LPMC_RR with current Biogeme optimization algorithm.")

    model = LPMC_RR(data_folder)

    res = {'time': [], 'LL': [], 'epochs': []}

    for i in range(20):

        results = model.biogeme.estimate()

        res['time'].append(model.biogeme.optimizationTime.total_seconds())
        res['LL'].append(results.data.logLike)
        res['epochs'].append(model.biogeme.numberOfIterations)

        with open('results/biogeme_bfgs.json', 'w') as outfile:
            json.dump(res, outfile)

        # Delete pickle files
        all_files = os.listdir()

        for item in all_files:
            if item.endswith(".pickle"):
                os.remove(item)

        print("{}/20 done!".format(i+1))
