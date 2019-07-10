"""
Simple MNL with Swissmetro
"""

import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.models as models
from biogeme.expressions import *
import numpy as np
import pandas as pd

class MNL:

    def __init__(self, data_folder=None, df=None):
        self.data_folder = data_folder
        self.df = df

        self.biogeme = None

        self.prep_db()

        self.build()

    def prep_db(self):

        if self.df is not None:
            self.database = db.Database('swissmetro', self.df)

        else:
            pandas = pd.read_table(self.data_folder + 'GEV_SM/swissmetro.dat')

            self.database = db.Database('swissmetro', pandas)

        for col in self.database.data.columns:
            exec("self.%s = Variable('%s')" % (col, col))

    def build(self):

        exclude = ((self.PURPOSE != 1) * (self.PURPOSE != 3) + (self.CHOICE == 0)) > 0
        self.database.remove(exclude)

        ASC_CAR = Beta('ASC_CAR', 0, None, None, 0, 'Car cte.')
        ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0, 'Train cte.')
        ASC_SM = Beta('ASC_SM', 0, None, None, 1, 'Swissmetro cte.')
        B_TIME = Beta('B_TIME', 0, None, None, 0, 'Travel time')
        B_COST = Beta('B_COST', 0, None, None, 0, 'Travel cost')

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

        self.x0 = np.zeros(4)

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
        self.V = {1: V1,
             2: V2,
             3: V3}

        # Associate the availability conditions with the alternatives
        CAR_AV_SP = DefineVariable('CAR_AV_SP', self.CAR_AV * (self.SP != 0), self.database)
        TRAIN_AV_SP = DefineVariable('TRAIN_AV_SP', self.TRAIN_AV * (self.SP != 0), self.database)

        self.av = {1: TRAIN_AV_SP,
              2: self.SM_AV,
              3: CAR_AV_SP}

        logprob = bioLogLogit(self.V, self.av, self.CHOICE)

        self.biogeme = bio.BIOGEME(self.database, logprob)
        self.biogeme.modelName = "01logit"
        self.biogeme.generateHtml = False

    def optimize(self, algo, **kwargs):

        if 'biogeme' in kwargs.keys():
            raise ValueError('Please remove biogeme from the kwargs.')

        params = {'biogeme': self.biogeme,
                  'full_size': len(self.biogeme.database.data)}

        for k in kwargs.keys():
            params[k] = kwargs[k]

        self.biogeme.database = self.database
        self.biogeme.theC.setData(self.biogeme.database.data)

        self.algo = algo(self.biogeme.calculateLikelihood, self.x0, **params)

        return self.algo.solve(maximize=True)

    def probas(self, betas=None):

        # The choice model is a logit, with availability conditions
        prob1 = models.logit(self.V, self.av, 1)
        prob2 = models.logit(self.V, self.av, 2)
        prob3 = models.logit(self.V, self.av, 3)

        simulate = {'Train': prob1,
                    'SM': prob2,
                    'Car': prob3}

        biogeme = bio.BIOGEME(self.database, simulate)
        biogeme.modelName = "01logit_simul"
        self.biogeme.generateHtml = False
        return biogeme.simulate(betas)






