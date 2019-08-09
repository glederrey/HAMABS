import sys
sys.path.append("../..")

import os
import numpy as np
import json

from algos import OptAlg

from models import LPMC_Full

data_folder = '../../../data/'


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


if __name__ == "__main__":

    if not os.path.exists('./results'):
        os.makedirs('./results')

    print("Train LPMC_Full_big with LS-ABS and hybrid-inv")

    model = LPMC_Full(data_folder, file='12_13_14.csv')

    ioa = OptAlg(alg_type='LS-ABS', direction='hybrid-inv')

    res = model.optimize(ioa, **{'verbose': False, 'nbr_epochs': 1000, 'batch': 1000})

    print(res)

    dumped = json.dumps(res, cls=NumpyEncoder)

    with open('results/res.json', 'w') as outfile:
        json.dump(dumped, outfile)
