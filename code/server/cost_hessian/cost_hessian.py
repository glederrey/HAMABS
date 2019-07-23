import sys
sys.path.append("../..")

import os
import time
import json
import numpy as np

from models import LPMC_Full
from models import LPMC_DrivingCost
from models import LPMC_RemoveRest
from models import Nested
from models import MNL

data_folder = '../../../data/'

if __name__ == '__main__':

    if not os.path.exists('./results'):
        os.makedirs('./results')

    models = [MNL, Nested, LPMC_DrivingCost, LPMC_RemoveRest, LPMC_Full]
    str_model = ['MNL', 'Nested', 'LPMC_DrivingCost', 'LPMC_RemoveRest', 'LPMC_Full']

    for model_constr, str_ in zip(models, str_model):
        print("Start testing Hessian for model {}".format(str_))

        if 'LPMC' in str_:
            model = model_constr(data_folder, '12_13_14.csv')
        else:
            model = model_constr(data_folder)

        df = model.biogeme.database.data

        n = len(model.biogeme.database.data)

        steps = 9
        if 'LPMC' in str_:
            steps = 19

        delta = int(n / steps)

        sizes = list(range(delta, n, delta))
        sizes[-1] = n

        tot_rep1 = 4
        tot_rep2 = 5

        tot_rep = tot_rep1 * tot_rep2

        res = {'size': sizes, 'time': []}
        for size in sizes:
            print("  Start for size {}. Total of {} iterations.".format(size, tot_rep))
            tmp = []
            count = 0
            for rep1 in range(tot_rep1):
                sample = df.sample(n=size, replace=False)

                model.biogeme.theC.setData(sample)

                for rep2 in range(tot_rep2):
                    print("    Iter {}/{}".format(count + 1, tot_rep))

                    start = time.time()
                    model.biogeme.calculateLikelihoodAndDerivatives(model.x0, hessian=True)
                    stop = time.time()

                    count += 1

                    tmp.append(stop - start)

            print("    Avg time obtained: {:.3f}\n".format(np.mean(tmp)))
            res['time'].append(tmp)

            with open('./results/' + str_ + '.json', 'w') as outfile:
                json.dump(res, outfile)

        print("\n")