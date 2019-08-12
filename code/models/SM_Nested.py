"""
Simple Nested Logit with Swissmetro
"""

import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.models as models
from biogeme.expressions import *
import numpy as np
import pandas as pd

class SM_Nested:

    def __init__(self, data_folder):
        self.data_folder = data_folder

        self.biogeme = None
        self.x0 = None

        self.prep_db()

        self.build()

    def prep_db(self):

        pandas = pd.read_table(self.data_folder + 'GEV_SM/swissmetro.dat')

        self.database = db.Database('swissmetro', pandas)

        for col in self.database.data.columns:
            exec("self.%s = Variable('%s')" % (col, col))

    def build(self):

        exclude = ((self.PURPOSE != 1) * (self.PURPOSE != 3) + (self.CHOICE == 0)) > 0
        self.database.remove(exclude)

        ASC_CAR = Beta('ASC_CAR', 0, -5, 5, 0, 'Car cte.')
        ASC_TRAIN = Beta('ASC_TRAIN', 0, -5, 5, 0, 'Train cte.')
        ASC_SM = Beta('ASC_SM', 0, None, None, 1, 'Swissmetro cte.')
        B_TIME = Beta('B_TIME', 0, -5, 5, 0, 'Travel time')
        B_COST = Beta('B_COST', 0, -5, 5, 0, 'Travel cost')

        MU = Beta('MU', 2.05, 1, 5, 0)

        SM_COST = self.SM_CO * (self.GA == 0)
        TRAIN_COST = self.TRAIN_CO * (self.GA == 0)

        TRAIN_TT_SCALED = DefineVariable('TRAIN_TT_SCALED', \
                                         self.TRAIN_TT / 100.0, self.database)
        TRAIN_COST_SCALED = DefineVariable('TRAIN_COST_SCALED', \
                                           TRAIN_COST / 100, self.database)
        SM_TT_SCALED = DefineVariable('SM_TT_SCALED', self.SM_TT / 100.0, self.database)
        SM_COST_SCALED = DefineVariable('SM_COST_SCALED', SM_COST / 100, self.database)
        CAR_TT_SCALED = DefineVariable('CAR_TT_SCALED', self.CAR_TT / 100, self.database)
        CAR_CO_SCALED = DefineVariable('CAR_CO_SCALED', self.CAR_CO / 100, self.database)

        V1 = ASC_TRAIN + \
             B_TIME * TRAIN_TT_SCALED + \
             B_COST * TRAIN_COST_SCALED
        V2 = ASC_SM + \
             B_TIME * SM_TT_SCALED + \
             B_COST * SM_COST_SCALED
        V3 = ASC_CAR + \
             B_TIME * CAR_TT_SCALED + \
             B_COST * CAR_CO_SCALED

        # Associate utility functions with the numbering of alternatives
        V = {1: V1,
             2: V2,
             3: V3}

        # Associate the availability conditions with the alternatives
        CAR_AV_SP = DefineVariable('CAR_AV_SP', self.CAR_AV * (self.SP != 0), self.database)
        TRAIN_AV_SP = DefineVariable('TRAIN_AV_SP', self.TRAIN_AV * (self.SP != 0), self.database)

        av = {1: TRAIN_AV_SP,
              2: self.SM_AV,
              3: CAR_AV_SP}

        # Definition of nests:
        # 1: nests parameter
        # 2: list of alternatives
        existing = MU, [1, 3]
        future = 1.0, [2]
        nests = existing, future

        # The choice model is a nested logit, with availability conditions
        logprob = models.lognested(V, av, nests, self.CHOICE)
        self.biogeme = bio.BIOGEME(self.database, logprob)
        self.biogeme.modelName = "SM_Nested"
        self.biogeme.generateHtml = False

        self.biogeme.theC.setData(self.database.data)

        self.x0 = self.biogeme.betaInitValues

    def optimize(self, algo, **kwargs):

        self.biogeme.database = self.database
        self.biogeme.theC.setData(self.biogeme.database.data)

        kwargs['bounds'] = self.biogeme.bounds

        algo.__prep__(self.x0, self.biogeme, **kwargs)

        return algo.solve(maximize=True)
