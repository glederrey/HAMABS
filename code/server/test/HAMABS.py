import sys
sys.path.append("../..")

import os

from algos import OptAlg

from models import MTMC_MNL

data_folder = '../../../data/'

if __name__ == "__main__":

    if not os.path.exists('./results'):
        os.makedirs('./results')

    print("Train MTMC_MNL with HAMABS")

    model = MTMC_MNL(data_folder)

    ioa = OptAlg(alg_type='LS-ABS', direction='hybrid-inv')

    tmp = model.optimize(ioa, **{'verbose': True, 'nbr_epochs': 1000, 'batch': 1000})
