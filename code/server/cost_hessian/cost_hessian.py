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

    # Go through each model to test all of them
    for model_constr, str_ in zip(models, str_model):
        print("Start testing Hessian for model {}".format(str_))

        # Load the LPMC models with the bigger file
        if 'LPMC' in str_:
            model = model_constr(data_folder, '12_13_14.csv')
        else:
            model = model_constr(data_folder)

        # Get the df
        df = model.biogeme.database.data

        # Length of the data
        n = len(model.biogeme.database.data)

        # Number of steps. Bigger for LPMC since more data
        steps = 10
        if 'LPMC' in str_:
            steps = 20

        # Delta and batch size to be tested
        delta = n / steps

        sizes = np.array(np.arange(delta, n + 1, delta))

        sizes = [int(round(val)) for val in sizes]

        # Number of repetitions (1 = new sample, 2 = same sample)
        tot_rep1 = 4
        tot_rep2 = 5

        tot_rep = tot_rep1 * tot_rep2

        # Go through each size to be tested
        res = {'size': sizes, 'time': {'func': [], 'hess': []}}
        for size in sizes:
            print("  Start for size {}. Total of {} iterations.".format(size, tot_rep))
            tmp_func = []
            tmp_hess = []

            count = 0
            # For each repetion 1, we resample the data
            for rep1 in range(tot_rep1):
                sample = df.sample(n=size, replace=False)

                model.biogeme.theC.setData(sample)

                # We test it rep2 times
                for rep2 in range(tot_rep2):
                    print("    Iter {}/{}".format(count + 1, tot_rep))

                    # Compute the hessian using biogeme
                    start1 = time.time()
                    model.biogeme.calculateLikelihoodAndDerivatives(model.x0, hessian=True)
                    stop1 = time.time()

                    delta1 = stop1 - start1

                    # Same function without the hessian
                    start2 = time.time()
                    model.biogeme.calculateLikelihoodAndDerivatives(model.x0, hessian=False)
                    stop2 = time.time()

                    delta2 = stop2 - start2

                    count += 1

                    # Add the times in tmp arrays
                    tmp_func.append(delta1)
                    tmp_hess.append(np.abs(delta1 - delta2))

            # Add the results in the dict
            print("    Avg time obtained: {:.3E}\n".format(np.mean(tmp_func)))
            res['time']['func'].append(tmp_func)
            print("    Avg delta obtained: {:.3E}\n".format(np.mean(tmp_hess)))
            res['time']['hess'].append(tmp_hess)


            # Save the dict (in case something goes wrong=
            with open('./results/' + str_ + '.json', 'w') as outfile:
                json.dump(res, outfile)

        print("\n")
