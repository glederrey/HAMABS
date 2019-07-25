"""
Middle MNL from Tim with 54 parameters
"""

import biogeme.database as db
import biogeme.biogeme as bio
from biogeme.expressions import *
import numpy as np
import pandas as pd

class LPMC_RemoveRest:

    def __init__(self, data_folder, file='12_13.csv'):
        self.data_folder = data_folder

        self.file = file

        self.biogeme = None

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
        B_TRAFFIC_DRIVING = Beta('B_TRAFFIC_DRIVING', 0, -10, 10, 0)

        B_COST_DRIVE = Beta('B_COST_DRIVE', 0, -10, 10, 0)
        B_COST_PT = Beta('B_COST_PT', 0, -10, 10, 0)

        B_TIME_PT_BUS = Beta('B_TIME_PT_BUS', 0, -10, 10, 0)
        B_TIME_PT_RAIL = Beta('B_TIME_PT_RAIL', 0, -10, 10, 0)
        B_TIME_PT_ACCESS = Beta('B_TIME_PT_ACCESS', 0, -10, 10, 0)
        B_TIME_PT_INT_WALK = Beta('B_TIME_PT_INT_WALK', 0, -10, 10, 0)
        B_TIME_PT_INT_WAIT = Beta('B_TIME_PT_INT_WAIT', 0, -10, 10, 0)

        B_PURPOSE_B_WALKING = Beta('B_PURPOSE_B_WALKING', 0, -10, 10, 1)
        B_PURPOSE_B_CYCLING = Beta('B_PURPOSE_B_CYCLING', 0, -10, 10, 0)
        B_PURPOSE_B_PT = Beta('B_PURPOSE_B_PT', 0, -10, 10, 0)
        B_PURPOSE_B_DRIVING = Beta('B_PURPOSE_B_DRIVING', 0, -10, 10, 0)
        B_PURPOSE_HBW_WALKING = Beta('B_PURPOSE_HBW_WALKING', 0, -10, 10, 1)
        B_PURPOSE_HBW_CYCLING = Beta('B_PURPOSE_HBW_CYCLING', 0, -10, 10, 0)
        B_PURPOSE_HBW_PT = Beta('B_PURPOSE_HBW_PT', 0, -10, 10, 0)
        B_PURPOSE_HBW_DRIVING = Beta('B_PURPOSE_HBW_DRIVING', 0, -10, 10, 0)
        B_PURPOSE_HBE_WALKING = Beta('B_PURPOSE_HBE_WALKING', 0, -10, 10, 1)
        # B_PURPOSE_HBE_CYCLING = Beta('B_PURPOSE_HBE_CYCLING',0,-10,10,0)
        B_PURPOSE_HBE_PT = Beta('B_PURPOSE_HBE_PT', 0, -10, 10, 0)
        B_PURPOSE_HBE_DRIVING = Beta('B_PURPOSE_HBE_DRIVING', 0, -10, 10, 0)
        B_PURPOSE_HBO_WALKING = Beta('B_PURPOSE_HBO_WALKING', 0, -10, 10, 1)
        B_PURPOSE_HBO_CYCLING = Beta('B_PURPOSE_HBO_CYCLING', 0, -10, 10, 0)
        B_PURPOSE_HBO_PT = Beta('B_PURPOSE_HBO_PT', 0, -10, 10, 0)
        # B_PURPOSE_HBO_DRIVING = Beta('B_PURPOSE_HBO_DRIVING',0,-10,10,0)

        B_VEHICLE_OWNERSHIP_1_WALKING = Beta('B_VEHICLE_OWNERSHIP_1_WALKING', 0, -10, 10, 1)
        B_VEHICLE_OWNERSHIP_CYCLING = Beta('B_VEHICLE_OWNERSHIP_CYCLING', 0, -10, 10, 0)
        B_VEHICLE_OWNERSHIP_1_PT = Beta('B_VEHICLE_OWNERSHIP_1_PT', 0, -10, 10, 0)
        B_VEHICLE_OWNERSHIP_1_DRIVING = Beta('B_VEHICLE_OWNERSHIP_1_DRIVING', 0, -10, 10, 0)
        B_VEHICLE_OWNERSHIP_2_WALKING = Beta('B_VEHICLE_OWNERSHIP_2_WALKING', 0, -10, 10, 1)
        B_VEHICLE_OWNERSHIP_2_PT = Beta('B_VEHICLE_OWNERSHIP_2_PT', 0, -10, 10, 0)
        B_VEHICLE_OWNERSHIP_2_DRIVING = Beta('B_VEHICLE_OWNERSHIP_2_DRIVING', 0, -10, 10, 0)

        B_DRIVING_LICENCE_WALKING = Beta('B_DRIVING_LICENCE_WALKING', 0, -10, 10, 1)
        B_DRIVING_LICENCE_CYCLING = Beta('B_DRIVING_LICENCE_CYCLING', 0, -10, 10, 0)
        B_DRIVING_LICENCE_PT = Beta('B_DRIVING_LICENCE_PT', 0, -10, 10, 0)
        B_DRIVING_LICENCE_DRIVING = Beta('B_DRIVING_LICENCE_DRIVING', 0, -10, 10, 0)

        B_FEMALE_WALKING = Beta('B_FEMALE_WALKING', 0, -10, 10, 1)
        B_FEMALE_CYCLING = Beta('B_FEMALE_CYCLING', 0, -10, 10, 0)
        B_FEMALE_PT = Beta('B_FEMALE_PT', 0, -10, 10, 0)
        B_FEMALE_DRIVING = Beta('B_FEMALE_DRIVING', 0, -10, 10, 0)

        B_WINTER_WALKING = Beta('B_WINTER_WALKING', 0, -10, 10, 1)
        B_WINTER_CYCLING = Beta('B_WINTER_CYCLING', 0, -10, 10, 0)
        # B_WINTER_PT = Beta('B_WINTER_PT',0,-10,10,0)
        B_WINTER_DRIVING = Beta('B_WINTER_DRIVING', 0, -10, 10, 0)

        B_DISTANCE_WALKING = Beta('B_DISTANCE_WALKING', 0, -10, 10, 1)
        B_DISTANCE_CYCLING = Beta('B_DISTANCE_CYCLING', 0, -10, 10, 0)
        B_DISTANCE_PT = Beta('B_DISTANCE_PT', 0, -10, 10, 0)
        B_DISTANCE_DRIVING = Beta('B_DISTANCE_DRIVING', 0, -10, 10, 0)

        B_AGE_CHILD_WALKING = Beta('B_AGE_CHILD_WALKING', 0, -10, 10, 1)
        # B_AGE_CHILD_CYCLING = Beta('B_AGE_CHILD_CYCLING',0,-10,10,0)
        B_AGE_CHILD_PT = Beta('B_AGE_CHILD_PT', 0, -10, 10, 0)
        B_AGE_CHILD_DRIVING = Beta('B_AGE_CHILD_DRIVING', 0, -10, 10, 0)
        B_AGE_PENSIONER_WALKING = Beta('B_AGE_PENSIONER_WALKING', 0, -10, 10, 1)
        B_AGE_PENSIONER_CYCLING = Beta('B_AGE_PENSIONER_CYCLING', 0, -10, 10, 0)
        B_AGE_PENSIONER_PT = Beta('B_AGE_PENSIONER_PT', 0, -10, 10, 0)
        B_AGE_PENSIONER_DRIVING = Beta('B_AGE_PENSIONER_DRIVING', 0, -10, 10, 0)

        B_DAY_WEEK_WALKING = Beta('B_DAY_WEEK_WALKING', 0, -10, 10, 1)
        # B_DAY_WEEK_CYCLING = Beta('B_DAY_WEEK_CYCLING',0,-10,10,0)
        B_DAY_WEEK_PT = Beta('B_DAY_WEEK_PT', 0, -10, 10, 0)
        B_DAY_WEEK_DRIVING = Beta('B_DAY_WEEK_DRIVING', 0, -10, 10, 0)
        B_DAY_SAT_WALKING = Beta('B_DAY_SAT_WALKING', 0, -10, 10, 1)
        B_DAY_SAT_CYCLING = Beta('B_DAY_SAT_CYCLING', 0, -10, 10, 0)
        B_DAY_SAT_PT = Beta('B_DAY_SAT_PT', 0, -10, 10, 0)
        # B_DAY_SAT_DRIVING = Beta('B_DAY_SAT_DRIVING',0,-10,10,0)

        # B_DEPARTURE_AM_PEAK_WALKING = Beta('B_DEPARTURE_AM_PEAK_WALKING',0,-10,10,1)
        # B_DEPARTURE_AM_PEAK_CYCLING = Beta('B_DEPARTURE_AM_PEAK_CYCLING',0,-10,10,0)
        # B_DEPARTURE_AM_PEAK_PT = Beta('B_DEPARTURE_AM_PEAK_PT',0,-10,10,0)
        # B_DEPARTURE_AM_PEAK_DRIVING = Beta('B_DEPARTURE_AM_PEAK_DRIVING',0,-10,10,0)
        B_DEPARTURE_PM_PEAK_WALKING = Beta('B_DEPARTURE_PM_PEAK_WALKING', 0, -10, 10, 1)
        B_DEPARTURE_PM_PEAK_CYCLING = Beta('B_DEPARTURE_PM_PEAK_CYCLING', 0, -10, 10, 0)
        B_DEPARTURE_PM_PEAK_PT = Beta('B_DEPARTURE_PM_PEAK_PT', 0, -10, 10, 0)
        B_DEPARTURE_PM_PEAK_DRIVING = Beta('B_DEPARTURE_PM_PEAK_DRIVING', 0, -10, 10, 0)
        B_DEPARTURE_INTER_PEAK_WALKING = Beta('B_DEPARTURE_INTER_PEAK_WALKING', 0, -10, 10, 1)
        B_DEPARTURE_INTER_PEAK_CYCLING = Beta('B_DEPARTURE_INTER_PEAK_CYCLING', 0, -10, 10, 0)
        # B_DEPARTURE_INTER_PEAK_PT = Beta('B_DEPARTURE_INTER_PEAK_PT',0,-10,10,0)
        B_DEPARTURE_INTER_PEAK_DRIVING = Beta('B_DEPARTURE_INTER_PEAK_DRIVING', 0, -10, 10, 0)

        # Defined variables

        co1 = DefineVariable('co1', self.car_ownership == 1, self.database)
        co2 = DefineVariable('co2', self.car_ownership == 2, self.database)
        weekday = DefineVariable('weekday', self.day_of_week < 6, self.database)
        saturday = DefineVariable('saturday', self.day_of_week == 6, self.database)
        child = DefineVariable('child', self.age < 18, self.database)
        pensioner = DefineVariable('pensioner', self.age > 64, self.database)
        winter = DefineVariable('winter', self.travel_month < 3 or self.travel_month == 12, self.database)
        # ampeak = DefineVariable('ampeak', self.start_time_linear<9.5 and self.start_time_linear>=6.5, self.database)
        pmpeak = DefineVariable('pmpeak', self.start_time_linear < 19.5 and self.start_time_linear >= 16.5, self.database)
        interpeak = DefineVariable('interpeak', self.start_time_linear < 16.5 and self.start_time_linear >= 9.5, self.database)
        distance_km = DefineVariable('distance_km', self.distance / 1000, self.database)
        drive_cost = DefineVariable('drive_cost', self.cost_driving_total * (self.car_ownership > 0), self.database)

        # Utility functions

        V1 = (ASC_WALKING +
              B_TIME_WALKING * self.dur_walking +
              B_PURPOSE_B_WALKING * self.purpose_B +
              B_PURPOSE_HBW_WALKING * self.purpose_HBW +
              B_PURPOSE_HBE_WALKING * self.purpose_HBE +
              B_PURPOSE_HBO_WALKING * self.purpose_HBO +
              B_VEHICLE_OWNERSHIP_1_WALKING * co1 +
              B_VEHICLE_OWNERSHIP_2_WALKING * co2 +
              B_FEMALE_WALKING * self.female +
              B_WINTER_WALKING * winter +
              B_AGE_CHILD_WALKING * child +
              B_AGE_PENSIONER_WALKING * pensioner +
              B_DRIVING_LICENCE_WALKING * self.driving_license +
              B_DAY_WEEK_WALKING * weekday +
              B_DAY_SAT_WALKING * saturday +
              # B_DEPARTURE_AM_PEAK_WALKING * ampeak +
              B_DEPARTURE_INTER_PEAK_WALKING * interpeak +
              B_DEPARTURE_PM_PEAK_WALKING * pmpeak +
              B_DISTANCE_WALKING * distance_km
              )

        V2 = (ASC_CYCLING +
              B_TIME_CYCLING * self.dur_cycling +
              B_PURPOSE_B_CYCLING * self.purpose_B +
              B_PURPOSE_HBW_CYCLING * self.purpose_HBW +
              # B_PURPOSE_HBE_CYCLING * purpose_HBE +
              B_PURPOSE_HBO_CYCLING * self.purpose_HBO +
              B_VEHICLE_OWNERSHIP_CYCLING * co1 +
              B_VEHICLE_OWNERSHIP_CYCLING * co2 +
              B_FEMALE_CYCLING * self.female +
              B_WINTER_CYCLING * winter +
              # B_AGE_CHILD_CYCLING * child +
              B_AGE_PENSIONER_CYCLING * pensioner +
              B_DRIVING_LICENCE_CYCLING * self.driving_license +
              # B_DAY_WEEK_CYCLING * weekday +
              B_DAY_SAT_CYCLING * saturday +
              # B_DEPARTURE_AM_PEAK_CYCLING * ampeak +
              B_DEPARTURE_INTER_PEAK_CYCLING * interpeak +
              B_DEPARTURE_PM_PEAK_CYCLING * pmpeak +
              B_DISTANCE_CYCLING * distance_km
              )

        V3 = (ASC_PT +
              B_COST_PT * self.cost_transit +
              B_TIME_PT_ACCESS * self.dur_access +
              B_TIME_PT_RAIL * self.dur_pt_rail +
              B_TIME_PT_BUS * self.dur_pt_bus +
              B_TIME_PT_INT_WAIT * self.dur_interchange_waiting +
              B_TIME_PT_INT_WALK * self.dur_interchange_walking +
              B_PURPOSE_B_PT * self.purpose_B +
              B_PURPOSE_HBW_PT * self.purpose_HBW +
              B_PURPOSE_HBE_PT * self.purpose_HBE +
              B_PURPOSE_HBO_PT * self.purpose_HBO +
              B_VEHICLE_OWNERSHIP_1_PT * co1 +
              B_VEHICLE_OWNERSHIP_2_PT * co2 +
              B_FEMALE_PT * self.female +
              # B_WINTER_PT * winter +
              B_AGE_CHILD_PT * child +
              B_AGE_PENSIONER_PT * pensioner +
              B_DRIVING_LICENCE_PT * self.driving_license +
              B_DAY_WEEK_PT * weekday +
              B_DAY_SAT_PT * saturday +
              # B_DEPARTURE_AM_PEAK_PT * ampeak +
              # B_DEPARTURE_INTER_PEAK_PT * interpeak +
              B_DEPARTURE_PM_PEAK_PT * pmpeak +
              B_DISTANCE_PT * distance_km
              )

        V4 = (ASC_DRIVING +
              B_TIME_DRIVING * self.dur_driving +
              B_COST_DRIVE * drive_cost +
              B_TRAFFIC_DRIVING * self.traffic_percent +
              B_PURPOSE_B_DRIVING * self.purpose_B +
              B_PURPOSE_HBW_DRIVING * self.purpose_HBW +
              B_PURPOSE_HBE_DRIVING * self.purpose_HBE +
              # B_PURPOSE_HBO_DRIVING * purpose_HBO +
              B_VEHICLE_OWNERSHIP_1_DRIVING * co1 +
              B_VEHICLE_OWNERSHIP_2_DRIVING * co2 +
              B_FEMALE_DRIVING * self.female +
              B_WINTER_DRIVING * winter +
              B_AGE_CHILD_DRIVING * child +
              B_AGE_PENSIONER_DRIVING * pensioner +
              B_DRIVING_LICENCE_DRIVING * self.driving_license +
              B_DAY_WEEK_DRIVING * weekday +
              # B_DAY_SAT_DRIVING * saturday +
              # B_DEPARTURE_AM_PEAK_DRIVING * ampeak +
              B_DEPARTURE_INTER_PEAK_DRIVING * interpeak +
              B_DEPARTURE_PM_PEAK_DRIVING * pmpeak +
              B_DISTANCE_DRIVING * distance_km
              )

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
        self.biogeme.modelName = "CLT_DrivingCostMNL"
        self.biogeme.generateHtml = False

        self.size_db = len(self.database.data)
        self.x0 = np.zeros(len(self.biogeme.betaInitValues))


    def optimize(self, algo, **kwargs):

        self.biogeme.database = self.database
        self.biogeme.theC.setData(self.biogeme.database.data)

        algo.__prep__(self.x0, self.biogeme, **kwargs)

        return algo.solve(maximize=True)
