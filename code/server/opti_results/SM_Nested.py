import sys
sys.path.append("../..")

import os
import numpy as np
import json

from algos import OptAlg

from models import SM_Nested

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

    model = SM_Nested(data_folder)

    ioa = OptAlg(alg_type='LS-ABS', direction='hybrid-inv')

    res = model.optimize(ioa, **{'verbose': True, 'nbr_epochs': 1000, 'batch': 1000})

    dumped = json.dumps(res, cls=NumpyEncoder)

    with open('results/SM_Nested.json', 'w') as outfile:
        json.dump(dumped, outfile)
