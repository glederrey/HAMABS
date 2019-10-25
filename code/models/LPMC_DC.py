"""
Simplest MNL from Tim with 13 parameters
"""

import biogeme.database as db
import biogeme.biogeme as bio
from biogeme.expressions import *
import numpy as np
import pandas as pd

class LPMC_DC:

    def __init__(self, data_folder, file='12_13.csv'):
        self.data_folder = data_folder

        self.file = file

        self.biogeme = None
        self.x0 = None

        self.prep_db()

        self.build()

    def prep_db(self):

        pandas = pd.read_csv(self.data_folder + 'LondonTravel/' + self.file)

        self.database = db.Database('LondonTravel', pandas)

        for col in self.database.data.columns:
            exec("self.%s = Variable('%s')" % (col, col))

    def build(self):

        # Parameters to be estimated

        ASC_WALKING = Beta('ASC_WALKING', 0, -10, 10, 1)
        ASC_CYCLING = Beta('ASC_CYCLING', 0, -10, 10, 0)
        ASC_PT = Beta('ASC_PT', 0, -10, 10, 0)
        ASC_DRIVING = Beta('ASC_DRIVING', 0, -10, 10, 0)
        B_TIME_WALKING = Beta('B_TIME_WALKING', 0, -10, 10, 0)
        B_TIME_CYCLING = Beta('B_TIME_CYCLING', 0, -10, 10, 0)
        B_TIME_DRIVING = Beta('B_TIME_DRIVING', 0, -10, 10, 0)
        B_COST_DRIVING = Beta('B_COST_DRIVING', 0, -10, 10, 0)
        B_COST_PT = Beta('B_COST_PT', 0, -10, 10, 0)
        B_TIME_PT_BUS = Beta('B_TIME_PT_BUS', 0, -10, 10, 0)
        B_TIME_PT_RAIL = Beta('B_TIME_PT_RAIL', 0, -10, 10, 0)
        B_TIME_PT_ACCESS = Beta('B_TIME_PT_ACCESS', 0, -10, 10, 0)
        # B_INT_WALK = Beta('B_INT_WALK',0,-10,10,0)
        B_TIME_PT_INT_WAIT = Beta('B_TIME_PT_INT_WAIT', 0, -10, 10, 0)
        B_TRAFFIC_DRIVING = Beta('B_TRAFFIC_DRIVING', 0, -10, 10, 0)

        # Utility functions

        V1 = (ASC_WALKING + B_TIME_WALKING * self.dur_walking)

        V2 = (ASC_CYCLING +
              B_TIME_CYCLING * self.dur_cycling)

        V3 = (ASC_PT +
              B_COST_PT * self.cost_transit +
              B_TIME_PT_ACCESS * self.dur_access +
              B_TIME_PT_RAIL * self.dur_pt_rail +
              B_TIME_PT_BUS * self.dur_pt_bus +
              B_TIME_PT_INT_WAIT * self.dur_interchange_waiting)

        V4 = (ASC_DRIVING +
              B_TIME_DRIVING * self.dur_driving +
              B_COST_DRIVING * (self.cost_driving + self.con_charge) +
              B_TRAFFIC_DRIVING * self.traffic_percent)

        # Associate utility functions with the numbering of alternatives
        V = {1: V1,
             2: V2,
             3: V3,
             4: V4}

        av = {1: 1,
              2: 1,
              3: 1,
              4: 1}

        # The choice model is a logit, with availability conditions
        logprob = bioLogLogit(V, av, self.tmode)

        self.biogeme = bio.BIOGEME(self.database, logprob)
        self.biogeme.modelName = "LPMC_MNL_DC"
        self.biogeme.generateHtml = False

        self.x0 = self.biogeme.betaInitValues

    def optimize(self, algo, **kwargs):

        self.biogeme.database = self.database
        self.biogeme.theC.setData(self.biogeme.database.data)

        algo.__prep__(self.x0, self.biogeme, **kwargs)

        return algo.solve(maximize=True)
