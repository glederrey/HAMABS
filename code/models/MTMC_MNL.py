"""
MTMC (Danalet & Mathys, 2018)
"""

import biogeme.database as db
import biogeme.biogeme as bio
from biogeme.expressions import *
import numpy as np
import pandas as pd

class MTMC_MNL:

    def __init__(self, data_folder):
        self.data_folder = data_folder

        self.biogeme = None
        self.x0 = None

        self.prep_db()

        self.build()

    def prep_db(self):

        df = pd.read_table(self.data_folder + 'MTMC/biogemeMTMC_2015_mobility_tools_cleaned.dat')

        # Change Boolean values to int
        df[['employed', 'hh_income_na', 'hh_income_less_than_4000', 'hh_income_4001_to_10000',
            'hh_income_more_than_10000', 'region_lake_geneva', 'region_espace_mittelland',
            'region_northern_switzerland', 'region_zurich', 'region_eastern_switzerland',
            'region_central_switzerland', 'region_tessin', 'public_transport_connection_quality_ARE_A',
            'public_transport_connection_quality_ARE_B', 'public_transport_connection_quality_ARE_C',
            'public_transport_connection_quality_ARE_D', 'public_transport_connection_quality_ARE_NA',
            'single_household', 'couple_without_children', 'couple_with_children', 'single_parent_with_children',
            'adults_with_elderly_care', 'not_family_household', 'full_time_work', 'part_time_work', 'studying',
            'inactive', 'active_without_known_work_percentage', 'no_post_school_educ',
            'secundary_education', 'tertiary_education', 'university', 'avail_HT', 'avail_car', 'male']] = df[
            ['employed', 'hh_income_na', 'hh_income_less_than_4000', 'hh_income_4001_to_10000',
             'hh_income_more_than_10000', 'region_lake_geneva', 'region_espace_mittelland',
             'region_northern_switzerland', 'region_zurich', 'region_eastern_switzerland',
             'region_central_switzerland', 'region_tessin', 'public_transport_connection_quality_ARE_A',
             'public_transport_connection_quality_ARE_B', 'public_transport_connection_quality_ARE_C',
             'public_transport_connection_quality_ARE_D', 'public_transport_connection_quality_ARE_NA',
             'single_household', 'couple_without_children', 'couple_with_children', 'single_parent_with_children',
             'adults_with_elderly_care', 'not_family_household', 'full_time_work', 'part_time_work', 'studying',
             'inactive', 'active_without_known_work_percentage', 'no_post_school_educ',
             'secundary_education', 'tertiary_education', 'university', 'avail_HT', 'avail_car', 'male']].astype(int)

        self.database = db.Database('mtmc', df)

        for col in self.database.data.columns:
            exec("self.%s = Variable('%s')" % (col, col))

    def build(self):

        # Variables

        ASC_CarAvail = Beta('ASC_CarAvail', 0, -10, 10, 0)
        ASC_CarAvail_GA_scaled = Beta('ASC_CarAvail_GA_scaled', 0, -10, 10, 0)
        ASC_CarAvail_HT = Beta('ASC_CarAvail_HT', 0, -10, 10, 0)
        ASC_CarAvail_HT_Verbund = Beta('ASC_CarAvail_HT_Verbund', 0, -10, 10, 0)
        ASC_CarAvail_Verbund = Beta('ASC_CarAvail_Verbund', 0, -10, 10, 0)
        ASC_GA_scaled = Beta('ASC_GA_scaled', 0, -10, 10, 0)
        ASC_HT = Beta('ASC_HT', 0, -10, 10, 0)
        ASC_HT_Verbund = Beta('ASC_HT_Verbund', 0, -10, 10, 0)
        ASC_No_Mobility_Tool = Beta('ASC_No_Mobility_Tool', 0, -10, 10, 1)
        ASC_Verbund = Beta('ASC_Verbund', 0, -10, 10, 0)
        B_AGE_CarAvail = Beta('B_AGE_CarAvail', 0, -10, 10, 1)
        B_AGE_CarAvail_GA = Beta('B_AGE_CarAvail_GA', 0, -10, 10, 1)
        B_AGE_CarAvail_HT = Beta('B_AGE_CarAvail_HT', 0, -10, 10, 1)
        B_AGE_CarAvail_HT_Verbund = Beta('B_AGE_CarAvail_HT_Verbund', 0, -10, 10, 1)
        B_AGE_CarAvail_Verbund = Beta('B_AGE_CarAvail_Verbund', 0, -10, 10, 1)
        B_AGE_GA = Beta('B_AGE_GA', 0, -10, 10, 1)
        B_AGE_HT = Beta('B_AGE_HT', 0, -10, 10, 1)
        B_AGE_HT_Verbund = Beta('B_AGE_HT_Verbund', 0, -10, 10, 1)
        B_AGE_SQUARE_CarAvail = Beta('B_AGE_SQUARE_CarAvail', 0, -10, 10, 1)
        B_AGE_SQUARE_CarAvail_GA = Beta('B_AGE_SQUARE_CarAvail_GA', 0, -10, 10, 1)
        B_AGE_SQUARE_CarAvail_HT = Beta('B_AGE_SQUARE_CarAvail_HT', 0, -10, 10, 1)
        B_AGE_SQUARE_CarAvail_HT_Verbund = Beta('B_AGE_SQUARE_CarAvail_HT_Verbund', 0, -10, 10, 1)
        B_AGE_SQUARE_CarAvail_Verbund = Beta('B_AGE_SQUARE_CarAvail_Verbund', 0, -10, 10, 1)
        B_AGE_SQUARE_GA = Beta('B_AGE_SQUARE_GA', 0, -10, 10, 1)
        B_AGE_SQUARE_HT = Beta('B_AGE_SQUARE_HT', 0, -10, 10, 1)
        B_AGE_SQUARE_HT_Verbund = Beta('B_AGE_SQUARE_HT_Verbund', 0, -10, 10, 1)
        B_AGE_SQUARE_Verbund = Beta('B_AGE_SQUARE_Verbund', 0, -10, 10, 1)
        B_AGE_TIME_MALE_CarAvail = Beta('B_AGE_TIME_MALE_CarAvail', 0, -10, 10, 0)
        B_AGE_TIME_MALE_CarAvail_GA = Beta('B_AGE_TIME_MALE_CarAvail_GA', 0, -10, 10, 1)
        B_AGE_TIME_MALE_CarAvail_HT = Beta('B_AGE_TIME_MALE_CarAvail_HT', 0, -10, 10, 0)
        B_AGE_TIME_MALE_CarAvail_HT_Verbund = Beta('B_AGE_TIME_MALE_CarAvail_HT_Verbund', 0, -10, 10, 0)
        B_AGE_TIME_MALE_CarAvail_Verbund = Beta('B_AGE_TIME_MALE_CarAvail_Verbund', 0, -10, 10, 0)
        B_AGE_TIME_MALE_GA = Beta('B_AGE_TIME_MALE_GA', 0, -10, 10, 0)
        B_AGE_TIME_MALE_HT = Beta('B_AGE_TIME_MALE_HT', 0, -10, 10, 0)
        B_AGE_TIME_MALE_HT_Verbund = Beta('B_AGE_TIME_MALE_HT_Verbund', 0, -10, 10, 0)
        B_AGE_TIME_MALE_Verbund = Beta('B_AGE_TIME_MALE_Verbund', 0, -10, 10, 0)
        B_AGE_Verbund = Beta('B_AGE_Verbund', 0, -10, 10, 1)
        B_EMPLOYED_CarAvail = Beta('B_EMPLOYED_CarAvail', 0, -10, 10, 1)
        B_EMPLOYED_CarAvail_GA = Beta('B_EMPLOYED_CarAvail_GA', 0, -10, 10, 1)
        B_EMPLOYED_CarAvail_HT = Beta('B_EMPLOYED_CarAvail_HT', 0, -10, 10, 1)
        B_EMPLOYED_CarAvail_HT_Verbund = Beta('B_EMPLOYED_CarAvail_HT_Verbund', 0, -10, 10, 1)
        B_EMPLOYED_CarAvail_Verbund = Beta('B_EMPLOYED_CarAvail_Verbund', 0, -10, 10, 1)
        B_EMPLOYED_GA = Beta('B_EMPLOYED_GA', 0, -10, 10, 1)
        B_EMPLOYED_HT = Beta('B_EMPLOYED_HT', 0, -10, 10, 1)
        B_EMPLOYED_HT_Verbund = Beta('B_EMPLOYED_HT_Verbund', 0, -10, 10, 1)
        B_EMPLOYED_Verbund = Beta('B_EMPLOYED_Verbund', 0, -10, 10, 1)
        B_HH_INCOME_4001_to_10000_CarAvail = Beta('B_HH_INCOME_4001_to_10000_CarAvail', 0, -10, 10, 0)
        B_HH_INCOME_4001_to_10000_CarAvail_GA = Beta('B_HH_INCOME_4001_to_10000_CarAvail_GA', 0, -10, 10, 0)
        B_HH_INCOME_4001_to_10000_CarAvail_HT = Beta('B_HH_INCOME_4001_to_10000_CarAvail_HT', 0, -10, 10, 0)
        B_HH_INCOME_4001_to_10000_CarAvail_HT_Verbund = Beta('B_HH_INCOME_4001_to_10000_CarAvail_HT_Verbund', 0, -10,
                                                             10, 0)
        B_HH_INCOME_4001_to_10000_CarAvail_Verbund = Beta('B_HH_INCOME_4001_to_10000_CarAvail_Verbund', 0, -10, 10, 0)
        B_HH_INCOME_4001_to_10000_GA = Beta('B_HH_INCOME_4001_to_10000_GA', 0, -10, 10, 0)
        B_HH_INCOME_4001_to_10000_HT = Beta('B_HH_INCOME_4001_to_10000_HT', 0, -10, 10, 0)
        B_HH_INCOME_4001_to_10000_HT_Verbund = Beta('B_HH_INCOME_4001_to_10000_HT_Verbund', 0, -10, 10, 0)
        B_HH_INCOME_4001_to_10000_Verbund = Beta('B_HH_INCOME_4001_to_10000_Verbund', 0, -10, 10, 0)
        B_HH_INCOME_LESS_THAN_4000_CarAvail = Beta('B_HH_INCOME_LESS_THAN_4000_CarAvail', 0, -10, 10, 1)
        B_HH_INCOME_LESS_THAN_4000_CarAvail_GA = Beta('B_HH_INCOME_LESS_THAN_4000_CarAvail_GA', 0, -10, 10, 1)
        B_HH_INCOME_LESS_THAN_4000_CarAvail_HT = Beta('B_HH_INCOME_LESS_THAN_4000_CarAvail_HT', 0, -10, 10, 1)
        B_HH_INCOME_LESS_THAN_4000_CarAvail_HT_Verbund = Beta('B_HH_INCOME_LESS_THAN_4000_CarAvail_HT_Verbund', 0, -10,
                                                              10, 1)
        B_HH_INCOME_LESS_THAN_4000_CarAvail_Verbund = Beta('B_HH_INCOME_LESS_THAN_4000_CarAvail_Verbund', 0, -10, 10, 1)
        B_HH_INCOME_LESS_THAN_4000_GA = Beta('B_HH_INCOME_LESS_THAN_4000_GA', 0, -10, 10, 1)
        B_HH_INCOME_LESS_THAN_4000_HT = Beta('B_HH_INCOME_LESS_THAN_4000_HT', 0, -10, 10, 1)
        B_HH_INCOME_LESS_THAN_4000_HT_Verbund = Beta('B_HH_INCOME_LESS_THAN_4000_HT_Verbund', 0, -10, 10, 1)
        B_HH_INCOME_LESS_THAN_4000_Verbund = Beta('B_HH_INCOME_LESS_THAN_4000_Verbund', 0, -10, 10, 1)
        B_HH_INCOME_MORE_THAN_10000_CarAvail = Beta('B_HH_INCOME_MORE_THAN_10000_CarAvail', 0, -10, 10, 0)
        B_HH_INCOME_MORE_THAN_10000_CarAvail_GA = Beta('B_HH_INCOME_MORE_THAN_10000_CarAvail_GA', 0, -10, 10, 0)
        B_HH_INCOME_MORE_THAN_10000_CarAvail_HT = Beta('B_HH_INCOME_MORE_THAN_10000_CarAvail_HT', 0, -10, 10, 0)
        B_HH_INCOME_MORE_THAN_10000_CarAvail_HT_Verbund = Beta('B_HH_INCOME_MORE_THAN_10000_CarAvail_HT_Verbund', 0,
                                                               -10, 10, 0)
        B_HH_INCOME_MORE_THAN_10000_CarAvail_Verbund = Beta('B_HH_INCOME_MORE_THAN_10000_CarAvail_Verbund', 0, -10, 10,
                                                            0)
        B_HH_INCOME_MORE_THAN_10000_GA = Beta('B_HH_INCOME_MORE_THAN_10000_GA', 0, -10, 10, 0)
        B_HH_INCOME_MORE_THAN_10000_HT = Beta('B_HH_INCOME_MORE_THAN_10000_HT', 0, -10, 10, 0)
        B_HH_INCOME_MORE_THAN_10000_HT_Verbund = Beta('B_HH_INCOME_MORE_THAN_10000_HT_Verbund', 0, -10, 10, 0)
        B_HH_INCOME_MORE_THAN_10000_Verbund = Beta('B_HH_INCOME_MORE_THAN_10000_Verbund', 0, -10, 10, 0)
        B_HH_INCOME_NA_CarAvail = Beta('B_HH_INCOME_NA_CarAvail', 0, -10, 10, 0)
        B_HH_INCOME_NA_CarAvail_GA = Beta('B_HH_INCOME_NA_CarAvail_GA', 0, -10, 10, 0)
        B_HH_INCOME_NA_CarAvail_HT = Beta('B_HH_INCOME_NA_CarAvail_HT', 0, -10, 10, 0)
        B_HH_INCOME_NA_CarAvail_HT_Verbund = Beta('B_HH_INCOME_NA_CarAvail_HT_Verbund', 0, -10, 10, 0)
        B_HH_INCOME_NA_CarAvail_Verbund = Beta('B_HH_INCOME_NA_CarAvail_Verbund', 0, -10, 10, 0)
        B_HH_INCOME_NA_GA = Beta('B_HH_INCOME_NA_GA', 0, -10, 10, 0)
        B_HH_INCOME_NA_HT = Beta('B_HH_INCOME_NA_HT', 0, -10, 10, 0)
        B_HH_INCOME_NA_HT_Verbund = Beta('B_HH_INCOME_NA_HT_Verbund', 0, -10, 10, 0)
        B_HH_INCOME_NA_Verbund = Beta('B_HH_INCOME_NA_Verbund', 0, -10, 10, 0)
        B_INHABITANTS_CarAvail = Beta('B_INHABITANTS_CarAvail', 0, -10, 10, 0)
        B_INHABITANTS_CarAvail_GA = Beta('B_INHABITANTS_CarAvail_GA', 0, -10, 10, 0)
        B_INHABITANTS_CarAvail_HT = Beta('B_INHABITANTS_CarAvail_HT', 0, -10, 10, 0)
        B_INHABITANTS_CarAvail_HT_Verbund = Beta('B_INHABITANTS_CarAvail_HT_Verbund', 0, -10, 10, 0)
        B_INHABITANTS_CarAvail_Verbund = Beta('B_INHABITANTS_CarAvail_Verbund', 0, -10, 10, 0)
        B_INHABITANTS_GA = Beta('B_INHABITANTS_GA', 0, -10, 10, 0)
        B_INHABITANTS_HT = Beta('B_INHABITANTS_HT', 0, -10, 10, 0)
        B_INHABITANTS_HT_Verbund = Beta('B_INHABITANTS_HT_Verbund', 0, -10, 10, 0)
        B_INHABITANTS_Verbund = Beta('B_INHABITANTS_Verbund', 0, -10, 10, 0)
        B_LOG_INHABITANTS_CarAvail = Beta('B_LOG_INHABITANTS_CarAvail', 0, -10, 10, 0)
        B_LOG_INHABITANTS_CarAvail_GA = Beta('B_LOG_INHABITANTS_CarAvail_GA', 0, -10, 10, 0)
        B_LOG_INHABITANTS_CarAvail_HT = Beta('B_LOG_INHABITANTS_CarAvail_HT', 0, -10, 10, 0)
        B_LOG_INHABITANTS_CarAvail_HT_Verbund = Beta('B_LOG_INHABITANTS_CarAvail_HT_Verbund', 0, -10, 10, 0)
        B_LOG_INHABITANTS_CarAvail_Verbund = Beta('B_LOG_INHABITANTS_CarAvail_Verbund', 0, -10, 10, 0)
        B_LOG_INHABITANTS_GA = Beta('B_LOG_INHABITANTS_GA', 0, -10, 10, 0)
        B_LOG_INHABITANTS_HT = Beta('B_LOG_INHABITANTS_HT', 0, -10, 10, 0)
        B_LOG_INHABITANTS_HT_Verbund = Beta('B_LOG_INHABITANTS_HT_Verbund', 0, -10, 10, 0)
        B_LOG_INHABITANTS_Verbund = Beta('B_LOG_INHABITANTS_Verbund', 0, -10, 10, 0)
        B_MALE_CarAvail = Beta('B_MALE_CarAvail', 0, -10, 10, 0)
        B_MALE_CarAvail_GA = Beta('B_MALE_CarAvail_GA', 0, -10, 10, 0)
        B_MALE_CarAvail_HT = Beta('B_MALE_CarAvail_HT', 0, -10, 10, 0)
        B_MALE_CarAvail_HT_Verbund = Beta('B_MALE_CarAvail_HT_Verbund', 0, -10, 10, 0)
        B_MALE_CarAvail_Verbund = Beta('B_MALE_CarAvail_Verbund', 0, -10, 10, 0)
        B_MALE_GA = Beta('B_MALE_GA', 0, -10, 10, 0)
        B_MALE_HT = Beta('B_MALE_HT', 0, -10, 10, 0)
        B_MALE_HT_Verbund = Beta('B_MALE_HT_Verbund', 0, -10, 10, 0)
        B_MALE_Verbund = Beta('B_MALE_Verbund', 0, -10, 10, 0)
        B_PT_QUALITY_A_CarAvail = Beta('B_PT_QUALITY_A_CarAvail', 0, -10, 10, 1)
        B_PT_QUALITY_A_CarAvail_GA = Beta('B_PT_QUALITY_A_CarAvail_GA', 0, -10, 10, 1)
        B_PT_QUALITY_A_CarAvail_HT = Beta('B_PT_QUALITY_A_CarAvail_HT', 0, -10, 10, 1)
        B_PT_QUALITY_A_CarAvail_HT_Verbund = Beta('B_PT_QUALITY_A_CarAvail_HT_Verbund', 0, -10, 10, 1)
        B_PT_QUALITY_A_CarAvail_Verbund = Beta('B_PT_QUALITY_A_CarAvail_Verbund', 0, -10, 10, 1)
        B_PT_QUALITY_A_GA = Beta('B_PT_QUALITY_A_GA', 0, -10, 10, 1)
        B_PT_QUALITY_A_HT = Beta('B_PT_QUALITY_A_HT', 0, -10, 10, 1)
        B_PT_QUALITY_A_HT_Verbund = Beta('B_PT_QUALITY_A_HT_Verbund', 0, -10, 10, 1)
        B_PT_QUALITY_A_Verbund = Beta('B_PT_QUALITY_A_Verbund', 0, -10, 10, 1)
        B_PT_QUALITY_B_CarAvail = Beta('B_PT_QUALITY_B_CarAvail', 0, -10, 10, 1)
        B_PT_QUALITY_B_CarAvail_GA = Beta('B_PT_QUALITY_B_CarAvail_GA', 0, -10, 10, 1)
        B_PT_QUALITY_B_CarAvail_HT = Beta('B_PT_QUALITY_B_CarAvail_HT', 0, -10, 10, 1)
        B_PT_QUALITY_B_CarAvail_HT_Verbund = Beta('B_PT_QUALITY_B_CarAvail_HT_Verbund', 0, -10, 10, 1)
        B_PT_QUALITY_B_CarAvail_Verbund = Beta('B_PT_QUALITY_B_CarAvail_Verbund', 0, -10, 10, 1)
        B_PT_QUALITY_B_GA = Beta('B_PT_QUALITY_B_GA', 0, -10, 10, 1)
        B_PT_QUALITY_B_HT = Beta('B_PT_QUALITY_B_HT', 0, -10, 10, 1)
        B_PT_QUALITY_B_HT_Verbund = Beta('B_PT_QUALITY_B_HT_Verbund', 0, -10, 10, 1)
        B_PT_QUALITY_B_Verbund = Beta('B_PT_QUALITY_B_Verbund', 0, -10, 10, 1)
        B_PT_QUALITY_C_CarAvail = Beta('B_PT_QUALITY_C_CarAvail', 0, -10, 10, 1)
        B_PT_QUALITY_C_CarAvail_GA = Beta('B_PT_QUALITY_C_CarAvail_GA', 0, -10, 10, 1)
        B_PT_QUALITY_C_CarAvail_HT = Beta('B_PT_QUALITY_C_CarAvail_HT', 0, -10, 10, 1)
        B_PT_QUALITY_C_CarAvail_HT_Verbund = Beta('B_PT_QUALITY_C_CarAvail_HT_Verbund', 0, -10, 10, 1)
        B_PT_QUALITY_C_CarAvail_Verbund = Beta('B_PT_QUALITY_C_CarAvail_Verbund', 0, -10, 10, 1)
        B_PT_QUALITY_C_GA = Beta('B_PT_QUALITY_C_GA', 0, -10, 10, 1)
        B_PT_QUALITY_C_HT = Beta('B_PT_QUALITY_C_HT', 0, -10, 10, 1)
        B_PT_QUALITY_C_HT_Verbund = Beta('B_PT_QUALITY_C_HT_Verbund', 0, -10, 10, 1)
        B_PT_QUALITY_C_Verbund = Beta('B_PT_QUALITY_C_Verbund', 0, -10, 10, 1)
        B_PT_QUALITY_D_CarAvail = Beta('B_PT_QUALITY_D_CarAvail', 0, -10, 10, 1)
        B_PT_QUALITY_D_CarAvail_GA = Beta('B_PT_QUALITY_D_CarAvail_GA', 0, -10, 10, 1)
        B_PT_QUALITY_D_CarAvail_HT = Beta('B_PT_QUALITY_D_CarAvail_HT', 0, -10, 10, 1)
        B_PT_QUALITY_D_CarAvail_HT_Verbund = Beta('B_PT_QUALITY_D_CarAvail_HT_Verbund', 0, -10, 10, 1)
        B_PT_QUALITY_D_CarAvail_Verbund = Beta('B_PT_QUALITY_D_CarAvail_Verbund', 0, -10, 10, 1)
        B_PT_QUALITY_D_GA = Beta('B_PT_QUALITY_D_GA', 0, -10, 10, 1)
        B_PT_QUALITY_D_HT = Beta('B_PT_QUALITY_D_HT', 0, -10, 10, 1)
        B_PT_QUALITY_D_HT_Verbund = Beta('B_PT_QUALITY_D_HT_Verbund', 0, -10, 10, 1)
        B_PT_QUALITY_D_Verbund = Beta('B_PT_QUALITY_D_Verbund', 0, -10, 10, 1)
        B_PT_QUALITY_NA_CarAvail = Beta('B_PT_QUALITY_NA_CarAvail', 0, -10, 10, 1)
        B_PT_QUALITY_NA_CarAvail_GA = Beta('B_PT_QUALITY_NA_CarAvail_GA', 0, -10, 10, 1)
        B_PT_QUALITY_NA_CarAvail_HT = Beta('B_PT_QUALITY_NA_CarAvail_HT', 0, -10, 10, 1)
        B_PT_QUALITY_NA_CarAvail_HT_Verbund = Beta('B_PT_QUALITY_NA_CarAvail_HT_Verbund', 0, -10, 10, 1)
        B_PT_QUALITY_NA_CarAvail_Verbund = Beta('B_PT_QUALITY_NA_CarAvail_Verbund', 0, -10, 10, 1)
        B_PT_QUALITY_NA_GA = Beta('B_PT_QUALITY_NA_GA', 0, -10, 10, 1)
        B_PT_QUALITY_NA_HT = Beta('B_PT_QUALITY_NA_HT', 0, -10, 10, 1)
        B_PT_QUALITY_NA_HT_Verbund = Beta('B_PT_QUALITY_NA_HT_Verbund', 0, -10, 10, 1)
        B_PT_QUALITY_NA_Verbund = Beta('B_PT_QUALITY_NA_Verbund', 0, -10, 10, 1)
        B_REGION_CENTRAL_SWITZERLAND_CarAvail = Beta('B_REGION_CENTRAL_SWITZERLAND_CarAvail', 0, -10, 10, 0)
        B_REGION_CENTRAL_SWITZERLAND_CarAvail_GA = Beta('B_REGION_CENTRAL_SWITZERLAND_CarAvail_GA', 0, -10, 10, 0)
        B_REGION_CENTRAL_SWITZERLAND_CarAvail_HT = Beta('B_REGION_CENTRAL_SWITZERLAND_CarAvail_HT', 0, -10, 10, 0)
        B_REGION_CENTRAL_SWITZERLAND_CarAvail_HT_Verbund = Beta('B_REGION_CENTRAL_SWITZERLAND_CarAvail_HT_Verbund', 0,
                                                                -10, 10, 0)
        B_REGION_CENTRAL_SWITZERLAND_CarAvail_Verbund = Beta('B_REGION_CENTRAL_SWITZERLAND_CarAvail_Verbund', 0, -10,
                                                             10, 0)
        B_REGION_CENTRAL_SWITZERLAND_GA = Beta('B_REGION_CENTRAL_SWITZERLAND_GA', 0, -10, 10, 0)
        B_REGION_CENTRAL_SWITZERLAND_HT = Beta('B_REGION_CENTRAL_SWITZERLAND_HT', 0, -10, 10, 0)
        B_REGION_CENTRAL_SWITZERLAND_HT_Verbund = Beta('B_REGION_CENTRAL_SWITZERLAND_HT_Verbund', 0, -10, 10, 0)
        B_REGION_CENTRAL_SWITZERLAND_Verbund = Beta('B_REGION_CENTRAL_SWITZERLAND_Verbund', 0, -10, 10, 0)
        B_REGION_EASTERN_SWITZERLAND_CarAvail = Beta('B_REGION_EASTERN_SWITZERLAND_CarAvail', 0, -10, 10, 0)
        B_REGION_EASTERN_SWITZERLAND_CarAvail_GA = Beta('B_REGION_EASTERN_SWITZERLAND_CarAvail_GA', 0, -10, 10, 0)
        B_REGION_EASTERN_SWITZERLAND_CarAvail_HT = Beta('B_REGION_EASTERN_SWITZERLAND_CarAvail_HT', 0, -10, 10, 0)
        B_REGION_EASTERN_SWITZERLAND_CarAvail_HT_Verbund = Beta('B_REGION_EASTERN_SWITZERLAND_CarAvail_HT_Verbund', 0,
                                                                -10, 10, 0)
        B_REGION_EASTERN_SWITZERLAND_CarAvail_Verbund = Beta('B_REGION_EASTERN_SWITZERLAND_CarAvail_Verbund', 0, -10,
                                                             10, 0)
        B_REGION_EASTERN_SWITZERLAND_GA = Beta('B_REGION_EASTERN_SWITZERLAND_GA', 0, -10, 10, 0)
        B_REGION_EASTERN_SWITZERLAND_HT = Beta('B_REGION_EASTERN_SWITZERLAND_HT', 0, -10, 10, 0)
        B_REGION_EASTERN_SWITZERLAND_HT_Verbund = Beta('B_REGION_EASTERN_SWITZERLAND_HT_Verbund', 0, -10, 10, 0)
        B_REGION_EASTERN_SWITZERLAND_Verbund = Beta('B_REGION_EASTERN_SWITZERLAND_Verbund', 0, -10, 10, 0)
        B_REGION_ESPACE_MITTELLAND_CarAvail = Beta('B_REGION_ESPACE_MITTELLAND_CarAvail', 0, -10, 10, 0)
        B_REGION_ESPACE_MITTELLAND_CarAvail_GA = Beta('B_REGION_ESPACE_MITTELLAND_CarAvail_GA', 0, -10, 10, 0)
        B_REGION_ESPACE_MITTELLAND_CarAvail_HT = Beta('B_REGION_ESPACE_MITTELLAND_CarAvail_HT', 0, -10, 10, 0)
        B_REGION_ESPACE_MITTELLAND_CarAvail_HT_Verbund = Beta('B_REGION_ESPACE_MITTELLAND_CarAvail_HT_Verbund', 0, -10,
                                                              10, 0)
        B_REGION_ESPACE_MITTELLAND_CarAvail_Verbund = Beta('B_REGION_ESPACE_MITTELLAND_CarAvail_Verbund', 0, -10, 10, 0)
        B_REGION_ESPACE_MITTELLAND_GA = Beta('B_REGION_ESPACE_MITTELLAND_GA', 0, -10, 10, 0)
        B_REGION_ESPACE_MITTELLAND_HT = Beta('B_REGION_ESPACE_MITTELLAND_HT', 0, -10, 10, 0)
        B_REGION_ESPACE_MITTELLAND_HT_Verbund = Beta('B_REGION_ESPACE_MITTELLAND_HT_Verbund', 0, -10, 10, 0)
        B_REGION_ESPACE_MITTELLAND_Verbund = Beta('B_REGION_ESPACE_MITTELLAND_Verbund', 0, -10, 10, 0)
        B_REGION_LAKE_GENEVA_CarAvail = Beta('B_REGION_LAKE_GENEVA_CarAvail', 0, -10, 10, 0)
        B_REGION_LAKE_GENEVA_CarAvail_GA = Beta('B_REGION_LAKE_GENEVA_CarAvail_GA', 0, -10, 10, 0)
        B_REGION_LAKE_GENEVA_CarAvail_HT = Beta('B_REGION_LAKE_GENEVA_CarAvail_HT', 0, -10, 10, 0)
        B_REGION_LAKE_GENEVA_CarAvail_HT_Verbund = Beta('B_REGION_LAKE_GENEVA_CarAvail_HT_Verbund', 0, -10, 10, 0)
        B_REGION_LAKE_GENEVA_CarAvail_Verbund = Beta('B_REGION_LAKE_GENEVA_CarAvail_Verbund', 0, -10, 10, 0)
        B_REGION_LAKE_GENEVA_GA = Beta('B_REGION_LAKE_GENEVA_GA', 0, -10, 10, 0)
        B_REGION_LAKE_GENEVA_HT = Beta('B_REGION_LAKE_GENEVA_HT', 0, -10, 10, 0)
        B_REGION_LAKE_GENEVA_HT_Verbund = Beta('B_REGION_LAKE_GENEVA_HT_Verbund', 0, -10, 10, 0)
        B_REGION_LAKE_GENEVA_Verbund = Beta('B_REGION_LAKE_GENEVA_Verbund', 0, -10, 10, 0)
        B_REGION_NORTHERN_SWITZERLAND_CarAvail = Beta('B_REGION_NORTHERN_SWITZERLAND_CarAvail', 0, -10, 10, 0)
        B_REGION_NORTHERN_SWITZERLAND_CarAvail_GA = Beta('B_REGION_NORTHERN_SWITZERLAND_CarAvail_GA', 0, -10, 10, 0)
        B_REGION_NORTHERN_SWITZERLAND_CarAvail_HT = Beta('B_REGION_NORTHERN_SWITZERLAND_CarAvail_HT', 0, -10, 10, 0)
        B_REGION_NORTHERN_SWITZERLAND_CarAvail_HT_Verbund = Beta('B_REGION_NORTHERN_SWITZERLAND_CarAvail_HT_Verbund', 0,
                                                                 -10, 10, 0)
        B_REGION_NORTHERN_SWITZERLAND_CarAvail_Verbund = Beta('B_REGION_NORTHERN_SWITZERLAND_CarAvail_Verbund', 0, -10,
                                                              10, 0)
        B_REGION_NORTHERN_SWITZERLAND_GA = Beta('B_REGION_NORTHERN_SWITZERLAND_GA', 0, -10, 10, 0)
        B_REGION_NORTHERN_SWITZERLAND_HT = Beta('B_REGION_NORTHERN_SWITZERLAND_HT', 0, -10, 10, 0)
        B_REGION_NORTHERN_SWITZERLAND_HT_Verbund = Beta('B_REGION_NORTHERN_SWITZERLAND_HT_Verbund', 0, -10, 10, 0)
        B_REGION_NORTHERN_SWITZERLAND_Verbund = Beta('B_REGION_NORTHERN_SWITZERLAND_Verbund', 0, -10, 10, 0)
        B_REGION_TESSIN_CarAvail = Beta('B_REGION_TESSIN_CarAvail', 0, -10, 10, 1)
        B_REGION_TESSIN_CarAvail_GA = Beta('B_REGION_TESSIN_CarAvail_GA', 0, -10, 10, 1)
        B_REGION_TESSIN_CarAvail_HT = Beta('B_REGION_TESSIN_CarAvail_HT', 0, -10, 10, 1)
        B_REGION_TESSIN_CarAvail_HT_Verbund = Beta('B_REGION_TESSIN_CarAvail_HT_Verbund', 0, -10, 10, 1)
        B_REGION_TESSIN_CarAvail_Verbund = Beta('B_REGION_TESSIN_CarAvail_Verbund', 0, -10, 10, 1)
        B_REGION_TESSIN_GA = Beta('B_REGION_TESSIN_GA', 0, -10, 10, 1)
        B_REGION_TESSIN_HT = Beta('B_REGION_TESSIN_HT', 0, -10, 10, 1)
        B_REGION_TESSIN_HT_Verbund = Beta('B_REGION_TESSIN_HT_Verbund', 0, -10, 10, 1)
        B_REGION_TESSIN_Verbund = Beta('B_REGION_TESSIN_Verbund', 0, -10, 10, 1)
        B_REGION_ZURICH_CarAvail = Beta('B_REGION_ZURICH_CarAvail', 0, -10, 10, 0)
        B_REGION_ZURICH_CarAvail_GA = Beta('B_REGION_ZURICH_CarAvail_GA', 0, -10, 10, 0)
        B_REGION_ZURICH_CarAvail_HT = Beta('B_REGION_ZURICH_CarAvail_HT', 0, -10, 10, 0)
        B_REGION_ZURICH_CarAvail_HT_Verbund = Beta('B_REGION_ZURICH_CarAvail_HT_Verbund', 0, -10, 10, 0)
        B_REGION_ZURICH_CarAvail_Verbund = Beta('B_REGION_ZURICH_CarAvail_Verbund', 0, -10, 10, 0)
        B_REGION_ZURICH_GA = Beta('B_REGION_ZURICH_GA', 0, -10, 10, 0)
        B_REGION_ZURICH_HT = Beta('B_REGION_ZURICH_HT', 0, -10, 10, 0)
        B_REGION_ZURICH_HT_Verbund = Beta('B_REGION_ZURICH_HT_Verbund', 0, -10, 10, 0)
        B_REGION_ZURICH_Verbund = Beta('B_REGION_ZURICH_Verbund', 0, -10, 10, 0)
        B_active_without_known_work_percentage_CarAvail = Beta('B_active_without_known_work_percentage_CarAvail', 0,
                                                               -10, 10, 1)
        B_active_without_known_work_percentage_CarAvail_GA = Beta('B_active_without_known_work_percentage_CarAvail_GA',
                                                                  0, -10, 10, 1)
        B_active_without_known_work_percentage_CarAvail_HT = Beta('B_active_without_known_work_percentage_CarAvail_HT',
                                                                  0, -10, 10, 1)
        B_active_without_known_work_percentage_CarAvail_HT_Verbund = Beta(
            'B_active_without_known_work_percentage_CarAvail_HT_Verbund', 0, -10, 10, 1)
        B_active_without_known_work_percentage_CarAvail_Verbund = Beta(
            'B_active_without_known_work_percentage_CarAvail_Verbund', 0, -10, 10, 1)
        B_active_without_known_work_percentage_GA = Beta('B_active_without_known_work_percentage_GA', 0, -10, 10, 1)
        B_active_without_known_work_percentage_HT = Beta('B_active_without_known_work_percentage_HT', 0, -10, 10, 1)
        B_active_without_known_work_percentage_HT_Verbund = Beta('B_active_without_known_work_percentage_HT_Verbund', 0,
                                                                 -10, 10, 1)
        B_active_without_known_work_percentage_Verbund = Beta('B_active_without_known_work_percentage_Verbund', 0, -10,
                                                              10, 1)
        B_adults_with_elderly_care_CarAvail = Beta('B_adults_with_elderly_care_CarAvail', 0, -10, 10, 1)
        B_adults_with_elderly_care_CarAvail_GA = Beta('B_adults_with_elderly_care_CarAvail_GA', 0, -10, 10, 1)
        B_adults_with_elderly_care_CarAvail_HT = Beta('B_adults_with_elderly_care_CarAvail_HT', 0, -10, 10, 1)
        B_adults_with_elderly_care_CarAvail_HT_Verbund = Beta('B_adults_with_elderly_care_CarAvail_HT_Verbund', 0, -10,
                                                              10, 1)
        B_adults_with_elderly_care_CarAvail_Verbund = Beta('B_adults_with_elderly_care_CarAvail_Verbund', 0, -10, 10, 1)
        B_adults_with_elderly_care_GA = Beta('B_adults_with_elderly_care_GA', 0, -10, 10, 1)
        B_adults_with_elderly_care_HT = Beta('B_adults_with_elderly_care_HT', 0, -10, 10, 1)
        B_adults_with_elderly_care_HT_Verbund = Beta('B_adults_with_elderly_care_HT_Verbund', 0, -10, 10, 1)
        B_adults_with_elderly_care_Verbund = Beta('B_adults_with_elderly_care_Verbund', 0, -10, 10, 1)
        B_age_16_18_GA = Beta('B_age_16_18_GA', 0, -10, 10, 0)
        B_age_16_18_HT = Beta('B_age_16_18_HT', 0, -10, 10, 0)
        B_age_16_18_HT_Verbund = Beta('B_age_16_18_HT_Verbund', 0, -10, 10, 0)
        B_age_16_18_Verbund = Beta('B_age_16_18_Verbund', 0, -10, 10, 0)
        B_age_18_20_CarAvail_GA = Beta('B_age_18_20_CarAvail_GA', 0, -10, 10, 0)
        B_age_18_20_CarAvail_HT = Beta('B_age_18_20_CarAvail_HT', 0, -10, 10, 0)
        B_age_18_20_GA = Beta('B_age_18_20_GA', 0, -10, 10, 0)
        B_age_18_20_HT = Beta('B_age_18_20_HT', 0, -10, 10, 0)
        B_age_18_20_HT_Verbund = Beta('B_age_18_20_HT_Verbund', 0, -10, 10, 0)
        B_age_18_20_Verbund = Beta('B_age_18_20_Verbund', 0, -10, 10, 0)
        B_age_18_25_CarAvail = Beta('B_age_18_25_CarAvail', 0, -10, 10, 0)
        B_age_18_25_CarAvail_HT_Verbund = Beta('B_age_18_25_CarAvail_HT_Verbund', 0, -10, 10, 0)
        B_age_18_25_CarAvail_Verbund = Beta('B_age_18_25_CarAvail_Verbund', 0, -10, 10, 0)
        B_age_20_25_CarAvail_GA = Beta('B_age_20_25_CarAvail_GA', 0, -10, 10, 0)
        B_age_20_25_CarAvail_HT = Beta('B_age_20_25_CarAvail_HT', 0, -10, 10, 0)
        B_age_20_25_HT = Beta('B_age_20_25_HT', 0, -10, 10, 0)
        B_age_20_25_HT_Verbund = Beta('B_age_20_25_HT_Verbund', 0, -10, 10, 0)
        B_age_20_25_Verbund = Beta('B_age_20_25_Verbund', 0, -10, 10, 0)
        B_age_20_45_GA = Beta('B_age_20_45_GA', 0, -10, 10, 0)
        B_age_25_45_HT_Verbund = Beta('B_age_25_45_HT_Verbund', 0, -10, 10, 0)
        B_age_25_45_Verbund = Beta('B_age_25_45_Verbund', 0, -10, 10, 0)
        B_age_25_65_CarAvail = Beta('B_age_25_65_CarAvail', 0, -10, 10, 0)
        B_age_25_65_CarAvail_GA = Beta('B_age_25_65_CarAvail_GA', 0, -10, 10, 0)
        B_age_25_65_CarAvail_HT = Beta('B_age_25_65_CarAvail_HT', 0, -10, 10, 0)
        B_age_25_65_CarAvail_HT_Verbund = Beta('B_age_25_65_CarAvail_HT_Verbund', 0, -10, 10, 0)
        B_age_25_65_CarAvail_Verbund = Beta('B_age_25_65_CarAvail_Verbund', 0, -10, 10, 0)
        B_age_25_65_HT = Beta('B_age_25_65_HT', 0, -10, 10, 0)
        B_age_45_65_GA = Beta('B_age_45_65_GA', 0, -10, 10, 0)
        B_age_45_65_HT_Verbund = Beta('B_age_45_65_HT_Verbund', 0, -10, 10, 0)
        B_age_45_65_Verbund = Beta('B_age_45_65_Verbund', 0, -10, 10, 0)
        B_age_65_and_more_CarAvail = Beta('B_age_65_and_more_CarAvail', 0, -10, 10, 0)
        B_age_65_and_more_CarAvail_GA = Beta('B_age_65_and_more_CarAvail_GA', 0, -10, 10, 0)
        B_age_65_and_more_CarAvail_HT = Beta('B_age_65_and_more_CarAvail_HT', 0, -10, 10, 0)
        B_age_65_and_more_CarAvail_HT_Verbund = Beta('B_age_65_and_more_CarAvail_HT_Verbund', 0, -10, 10, 0)
        B_age_65_and_more_CarAvail_Verbund = Beta('B_age_65_and_more_CarAvail_Verbund', 0, -10, 10, 0)
        B_age_65_and_more_GA = Beta('B_age_65_and_more_GA', 0, -10, 10, 0)
        B_age_65_and_more_HT = Beta('B_age_65_and_more_HT', 0, -10, 10, 0)
        B_age_65_and_more_HT_Verbund = Beta('B_age_65_and_more_HT_Verbund', 0, -10, 10, 0)
        B_age_65_and_more_Verbund = Beta('B_age_65_and_more_Verbund', 0, -10, 10, 0)
        B_age_6_16_GA = Beta('B_age_6_16_GA', 0, -10, 10, 0)
        B_age_6_16_Verbund = Beta('B_age_6_16_Verbund', 0, -10, 10, 0)
        B_couple_with_children_CarAvail = Beta('B_couple_with_children_CarAvail', 0, -10, 10, 0)
        B_couple_with_children_CarAvail_GA = Beta('B_couple_with_children_CarAvail_GA', 0, -10, 10, 0)
        B_couple_with_children_CarAvail_HT = Beta('B_couple_with_children_CarAvail_HT', 0, -10, 10, 0)
        B_couple_with_children_CarAvail_HT_Verbund = Beta('B_couple_with_children_CarAvail_HT_Verbund', 0, -10, 10, 0)
        B_couple_with_children_CarAvail_Verbund = Beta('B_couple_with_children_CarAvail_Verbund', 0, -10, 10, 0)
        B_couple_with_children_GA = Beta('B_couple_with_children_GA', 0, -10, 10, 0)
        B_couple_with_children_HT = Beta('B_couple_with_children_HT', 0, -10, 10, 0)
        B_couple_with_children_HT_Verbund = Beta('B_couple_with_children_HT_Verbund', 0, -10, 10, 0)
        B_couple_with_children_Verbund = Beta('B_couple_with_children_Verbund', 0, -10, 10, 0)
        B_couple_without_children_CarAvail = Beta('B_couple_without_children_CarAvail', 0, -10, 10, 0)
        B_couple_without_children_CarAvail_GA = Beta('B_couple_without_children_CarAvail_GA', 0, -10, 10, 0)
        B_couple_without_children_CarAvail_HT = Beta('B_couple_without_children_CarAvail_HT', 0, -10, 10, 0)
        B_couple_without_children_CarAvail_HT_Verbund = Beta('B_couple_without_children_CarAvail_HT_Verbund', 0, -10,
                                                             10, 0)
        B_couple_without_children_CarAvail_Verbund = Beta('B_couple_without_children_CarAvail_Verbund', 0, -10, 10, 0)
        B_couple_without_children_GA = Beta('B_couple_without_children_GA', 0, -10, 10, 0)
        B_couple_without_children_HT = Beta('B_couple_without_children_HT', 0, -10, 10, 0)
        B_couple_without_children_HT_Verbund = Beta('B_couple_without_children_HT_Verbund', 0, -10, 10, 0)
        B_couple_without_children_Verbund = Beta('B_couple_without_children_Verbund', 0, -10, 10, 0)
        B_full_time_work_CarAvail = Beta('B_full_time_work_CarAvail', 0, -10, 10, 1)
        B_full_time_work_CarAvail_GA = Beta('B_full_time_work_CarAvail_GA', 0, -10, 10, 1)
        B_full_time_work_CarAvail_HT = Beta('B_full_time_work_CarAvail_HT', 0, -10, 10, 1)
        B_full_time_work_CarAvail_HT_Verbund = Beta('B_full_time_work_CarAvail_HT_Verbund', 0, -10, 10, 1)
        B_full_time_work_CarAvail_Verbund = Beta('B_full_time_work_CarAvail_Verbund', 0, -10, 10, 1)
        B_full_time_work_GA = Beta('B_full_time_work_GA', 0, -10, 10, 1)
        B_full_time_work_HT = Beta('B_full_time_work_HT', 0, -10, 10, 1)
        B_full_time_work_HT_Verbund = Beta('B_full_time_work_HT_Verbund', 0, -10, 10, 1)
        B_full_time_work_Verbund = Beta('B_full_time_work_Verbund', 0, -10, 10, 1)
        B_inactive_CarAvail = Beta('B_inactive_CarAvail', 0, -10, 10, 0)
        B_inactive_CarAvail_GA = Beta('B_inactive_CarAvail_GA', 0, -10, 10, 0)
        B_inactive_CarAvail_HT = Beta('B_inactive_CarAvail_HT', 0, -10, 10, 0)
        B_inactive_CarAvail_HT_Verbund = Beta('B_inactive_CarAvail_HT_Verbund', 0, -10, 10, 0)
        B_inactive_CarAvail_Verbund = Beta('B_inactive_CarAvail_Verbund', 0, -10, 10, 0)
        B_inactive_GA = Beta('B_inactive_GA', 0, -10, 10, 0)
        B_inactive_HT = Beta('B_inactive_HT', 0, -10, 10, 0)
        B_inactive_HT_Verbund = Beta('B_inactive_HT_Verbund', 0, -10, 10, 0)
        B_inactive_Verbund = Beta('B_inactive_Verbund', 0, -10, 10, 0)
        B_no_post_school_educ_CarAvail = Beta('B_no_post_school_educ_CarAvail', 0, -10, 10, 1)
        B_no_post_school_educ_CarAvail_GA = Beta('B_no_post_school_educ_CarAvail_GA', 0, -10, 10, 1)
        B_no_post_school_educ_CarAvail_HT = Beta('B_no_post_school_educ_CarAvail_HT', 0, -10, 10, 1)
        B_no_post_school_educ_CarAvail_HT_Verbund = Beta('B_no_post_school_educ_CarAvail_HT_Verbund', 0, -10, 10, 1)
        B_no_post_school_educ_CarAvail_Verbund = Beta('B_no_post_school_educ_CarAvail_Verbund', 0, -10, 10, 1)
        B_no_post_school_educ_GA = Beta('B_no_post_school_educ_GA', 0, -10, 10, 1)
        B_no_post_school_educ_HT = Beta('B_no_post_school_educ_HT', 0, -10, 10, 1)
        B_no_post_school_educ_HT_Verbund = Beta('B_no_post_school_educ_HT_Verbund', 0, -10, 10, 1)
        B_no_post_school_educ_Verbund = Beta('B_no_post_school_educ_Verbund', 0, -10, 10, 1)
        B_not_family_household_CarAvail = Beta('B_not_family_household_CarAvail', 0, -10, 10, 1)
        B_not_family_household_CarAvail_GA = Beta('B_not_family_household_CarAvail_GA', 0, -10, 10, 1)
        B_not_family_household_CarAvail_HT = Beta('B_not_family_household_CarAvail_HT', 0, -10, 10, 1)
        B_not_family_household_CarAvail_HT_Verbund = Beta('B_not_family_household_CarAvail_HT_Verbund', 0, -10, 10, 1)
        B_not_family_household_CarAvail_Verbund = Beta('B_not_family_household_CarAvail_Verbund', 0, -10, 10, 1)
        B_not_family_household_GA = Beta('B_not_family_household_GA', 0, -10, 10, 1)
        B_not_family_household_HT = Beta('B_not_family_household_HT', 0, -10, 10, 1)
        B_not_family_household_HT_Verbund = Beta('B_not_family_household_HT_Verbund', 0, -10, 10, 1)
        B_not_family_household_Verbund = Beta('B_not_family_household_Verbund', 0, -10, 10, 1)
        B_part_time_work_CarAvail = Beta('B_part_time_work_CarAvail', 0, -10, 10, 0)
        B_part_time_work_CarAvail_GA = Beta('B_part_time_work_CarAvail_GA', 0, -10, 10, 0)
        B_part_time_work_CarAvail_HT = Beta('B_part_time_work_CarAvail_HT', 0, -10, 10, 0)
        B_part_time_work_CarAvail_HT_Verbund = Beta('B_part_time_work_CarAvail_HT_Verbund', 0, -10, 10, 0)
        B_part_time_work_CarAvail_Verbund = Beta('B_part_time_work_CarAvail_Verbund', 0, -10, 10, 0)
        B_part_time_work_GA = Beta('B_part_time_work_GA', 0, -10, 10, 0)
        B_part_time_work_HT = Beta('B_part_time_work_HT', 0, -10, 10, 0)
        B_part_time_work_HT_Verbund = Beta('B_part_time_work_HT_Verbund', 0, -10, 10, 0)
        B_part_time_work_Verbund = Beta('B_part_time_work_Verbund', 0, -10, 10, 0)
        B_secundary_education_CarAvail = Beta('B_secundary_education_CarAvail', 0, -10, 10, 0)
        B_secundary_education_CarAvail_GA = Beta('B_secundary_education_CarAvail_GA', 0, -10, 10, 0)
        B_secundary_education_CarAvail_HT = Beta('B_secundary_education_CarAvail_HT', 0, -10, 10, 0)
        B_secundary_education_CarAvail_HT_Verbund = Beta('B_secundary_education_CarAvail_HT_Verbund', 0, -10, 10, 0)
        B_secundary_education_CarAvail_Verbund = Beta('B_secundary_education_CarAvail_Verbund', 0, -10, 10, 0)
        B_secundary_education_GA = Beta('B_secundary_education_GA', 0, -10, 10, 0)
        B_secundary_education_HT = Beta('B_secundary_education_HT', 0, -10, 10, 0)
        B_secundary_education_HT_Verbund = Beta('B_secundary_education_HT_Verbund', 0, -10, 10, 0)
        B_secundary_education_Verbund = Beta('B_secundary_education_Verbund', 0, -10, 10, 0)
        B_single_household_CarAvail = Beta('B_single_household_CarAvail', 0, -10, 10, 1)
        B_single_household_CarAvail_GA = Beta('B_single_household_CarAvail_GA', 0, -10, 10, 1)
        B_single_household_CarAvail_HT = Beta('B_single_household_CarAvail_HT', 0, -10, 10, 1)
        B_single_household_CarAvail_HT_Verbund = Beta('B_single_household_CarAvail_HT_Verbund', 0, -10, 10, 1)
        B_single_household_CarAvail_Verbund = Beta('B_single_household_CarAvail_Verbund', 0, -10, 10, 1)
        B_single_household_GA = Beta('B_single_household_GA', 0, -10, 10, 1)
        B_single_household_HT = Beta('B_single_household_HT', 0, -10, 10, 1)
        B_single_household_HT_Verbund = Beta('B_single_household_HT_Verbund', 0, -10, 10, 1)
        B_single_household_Verbund = Beta('B_single_household_Verbund', 0, -10, 10, 1)
        B_single_parent_with_children_CarAvail = Beta('B_single_parent_with_children_CarAvail', 0, -10, 10, 0)
        B_single_parent_with_children_CarAvail_GA = Beta('B_single_parent_with_children_CarAvail_GA', 0, -10, 10, 0)
        B_single_parent_with_children_CarAvail_HT = Beta('B_single_parent_with_children_CarAvail_HT', 0, -10, 10, 0)
        B_single_parent_with_children_CarAvail_HT_Verbund = Beta('B_single_parent_with_children_CarAvail_HT_Verbund', 0,
                                                                 -10, 10, 0)
        B_single_parent_with_children_CarAvail_Verbund = Beta('B_single_parent_with_children_CarAvail_Verbund', 0, -10,
                                                              10, 0)
        B_single_parent_with_children_GA = Beta('B_single_parent_with_children_GA', 0, -10, 10, 0)
        B_single_parent_with_children_HT = Beta('B_single_parent_with_children_HT', 0, -10, 10, 0)
        B_single_parent_with_children_HT_Verbund = Beta('B_single_parent_with_children_HT_Verbund', 0, -10, 10, 0)
        B_single_parent_with_children_Verbund = Beta('B_single_parent_with_children_Verbund', 0, -10, 10, 0)
        B_studying_CarAvail = Beta('B_studying_CarAvail', 0, -10, 10, 0)
        B_studying_CarAvail_GA = Beta('B_studying_CarAvail_GA', 0, -10, 10, 0)
        B_studying_CarAvail_HT = Beta('B_studying_CarAvail_HT', 0, -10, 10, 0)
        B_studying_CarAvail_HT_Verbund = Beta('B_studying_CarAvail_HT_Verbund', 0, -10, 10, 0)
        B_studying_CarAvail_Verbund = Beta('B_studying_CarAvail_Verbund', 0, -10, 10, 0)
        B_studying_GA = Beta('B_studying_GA', 0, -10, 10, 0)
        B_studying_HT = Beta('B_studying_HT', 0, -10, 10, 0)
        B_studying_HT_Verbund = Beta('B_studying_HT_Verbund', 0, -10, 10, 0)
        B_studying_Verbund = Beta('B_studying_Verbund', 0, -10, 10, 0)
        B_tertiary_education_CarAvail = Beta('B_tertiary_education_CarAvail', 0, -10, 10, 0)
        B_tertiary_education_CarAvail_GA = Beta('B_tertiary_education_CarAvail_GA', 0, -10, 10, 0)
        B_tertiary_education_CarAvail_HT = Beta('B_tertiary_education_CarAvail_HT', 0, -10, 10, 0)
        B_tertiary_education_CarAvail_HT_Verbund = Beta('B_tertiary_education_CarAvail_HT_Verbund', 0, -10, 10, 0)
        B_tertiary_education_CarAvail_Verbund = Beta('B_tertiary_education_CarAvail_Verbund', 0, -10, 10, 0)
        B_tertiary_education_GA = Beta('B_tertiary_education_GA', 0, -10, 10, 0)
        B_tertiary_education_HT = Beta('B_tertiary_education_HT', 0, -10, 10, 0)
        B_tertiary_education_HT_Verbund = Beta('B_tertiary_education_HT_Verbund', 0, -10, 10, 0)
        B_tertiary_education_Verbund = Beta('B_tertiary_education_Verbund', 0, -10, 10, 0)
        B_university_CarAvail = Beta('B_university_CarAvail', 0, -10, 10, 0)
        B_university_CarAvail_GA = Beta('B_university_CarAvail_GA', 0, -10, 10, 0)
        B_university_CarAvail_HT = Beta('B_university_CarAvail_HT', 0, -10, 10, 0)
        B_university_CarAvail_HT_Verbund = Beta('B_university_CarAvail_HT_Verbund', 0, -10, 10, 0)
        B_university_CarAvail_Verbund = Beta('B_university_CarAvail_Verbund', 0, -10, 10, 0)
        B_university_GA = Beta('B_university_GA', 0, -10, 10, 0)
        B_university_HT = Beta('B_university_HT', 0, -10, 10, 0)
        B_university_HT_Verbund = Beta('B_university_HT_Verbund', 0, -10, 10, 0)
        B_university_Verbund = Beta('B_university_Verbund', 0, -10, 10, 0)

        __CarAvail = ASC_CarAvail * self.one + B_MALE_CarAvail * self.male + B_AGE_CarAvail * self.age + B_AGE_SQUARE_CarAvail * self.age_square_scaled + B_age_18_25_CarAvail * self.age_18_25 + B_age_25_65_CarAvail * self.age_25_65 + B_age_65_and_more_CarAvail * self.age_65_and_more + B_EMPLOYED_CarAvail * self.employed + B_AGE_TIME_MALE_CarAvail * self.age_time_male_scaled + B_HH_INCOME_NA_CarAvail * self.hh_income_na + B_HH_INCOME_LESS_THAN_4000_CarAvail * self.hh_income_less_than_4000 + B_HH_INCOME_4001_to_10000_CarAvail * self.hh_income_4001_to_10000 + B_HH_INCOME_MORE_THAN_10000_CarAvail * self.hh_income_more_than_10000 + B_REGION_LAKE_GENEVA_CarAvail * self.region_lake_geneva + B_REGION_ESPACE_MITTELLAND_CarAvail * self.region_espace_mittelland + B_REGION_NORTHERN_SWITZERLAND_CarAvail * self.region_northern_switzerland + B_REGION_ZURICH_CarAvail * self.region_zurich + B_REGION_EASTERN_SWITZERLAND_CarAvail * self.region_eastern_switzerland + B_REGION_CENTRAL_SWITZERLAND_CarAvail * self.region_central_switzerland + B_REGION_TESSIN_CarAvail * self.region_tessin + B_INHABITANTS_CarAvail * self.pop_valid + B_LOG_INHABITANTS_CarAvail * self.log_pop_valid + B_PT_QUALITY_A_CarAvail * self.public_transport_connection_quality_ARE_A + B_PT_QUALITY_B_CarAvail * self.public_transport_connection_quality_ARE_B + B_PT_QUALITY_C_CarAvail * self.public_transport_connection_quality_ARE_C + B_PT_QUALITY_D_CarAvail * self.public_transport_connection_quality_ARE_C + B_PT_QUALITY_NA_CarAvail * self.public_transport_connection_quality_ARE_NA + B_single_household_CarAvail * self.single_household + B_couple_without_children_CarAvail * self.couple_without_children + B_couple_with_children_CarAvail * self.couple_with_children + B_single_parent_with_children_CarAvail * self.single_parent_with_children + B_adults_with_elderly_care_CarAvail * self.adults_with_elderly_care + B_not_family_household_CarAvail * self.not_family_household + B_full_time_work_CarAvail * self.full_time_work + B_part_time_work_CarAvail * self.part_time_work + B_studying_CarAvail * self.studying + B_inactive_CarAvail * self.inactive + B_active_without_known_work_percentage_CarAvail * self.active_without_known_work_percentage + B_no_post_school_educ_CarAvail * self.no_post_school_educ + B_secundary_education_CarAvail * self.secundary_education + B_tertiary_education_CarAvail * self.tertiary_education + B_university_CarAvail * self.university
        __CarAvail_GA = ASC_CarAvail_GA_scaled * self.ten + B_MALE_CarAvail_GA * self.male + B_AGE_CarAvail_GA * self.age + B_age_18_20_CarAvail_GA * self.age_18_20 + B_age_20_25_CarAvail_GA * self.age_20_25 + B_age_25_65_CarAvail_GA * self.age_25_65 + B_age_65_and_more_CarAvail_GA * self.age_65_and_more + B_AGE_SQUARE_CarAvail_GA * self.age_square_scaled + B_EMPLOYED_CarAvail_GA * self.employed + B_AGE_TIME_MALE_CarAvail_GA * self.age_time_male_scaled + B_HH_INCOME_NA_CarAvail_GA * self.hh_income_na + B_HH_INCOME_LESS_THAN_4000_CarAvail_GA * self.hh_income_less_than_4000 + B_HH_INCOME_4001_to_10000_CarAvail_GA * self.hh_income_4001_to_10000 + B_HH_INCOME_MORE_THAN_10000_CarAvail_GA * self.hh_income_more_than_10000 + B_REGION_LAKE_GENEVA_CarAvail_GA * self.region_lake_geneva + B_REGION_ESPACE_MITTELLAND_CarAvail_GA * self.region_espace_mittelland + B_REGION_NORTHERN_SWITZERLAND_CarAvail_GA * self.region_northern_switzerland + B_REGION_ZURICH_CarAvail_GA * self.region_zurich + B_REGION_EASTERN_SWITZERLAND_CarAvail_GA * self.region_eastern_switzerland + B_REGION_CENTRAL_SWITZERLAND_CarAvail_GA * self.region_central_switzerland + B_REGION_TESSIN_CarAvail_GA * self.region_tessin + B_INHABITANTS_CarAvail_GA * self.pop_valid + B_LOG_INHABITANTS_CarAvail_GA * self.log_pop_valid + B_PT_QUALITY_A_CarAvail_GA * self.public_transport_connection_quality_ARE_A + B_PT_QUALITY_B_CarAvail_GA * self.public_transport_connection_quality_ARE_B + B_PT_QUALITY_C_CarAvail_GA * self.public_transport_connection_quality_ARE_C + B_PT_QUALITY_D_CarAvail_GA * self.public_transport_connection_quality_ARE_C + B_PT_QUALITY_NA_CarAvail_GA * self.public_transport_connection_quality_ARE_NA + B_single_household_CarAvail_GA * self.single_household + B_couple_without_children_CarAvail_GA * self.couple_without_children + B_couple_with_children_CarAvail_GA * self.couple_with_children + B_single_parent_with_children_CarAvail_GA * self.single_parent_with_children + B_adults_with_elderly_care_CarAvail_GA * self.adults_with_elderly_care + B_not_family_household_CarAvail_GA * self.not_family_household + B_full_time_work_CarAvail_GA * self.full_time_work + B_part_time_work_CarAvail_GA * self.part_time_work + B_studying_CarAvail_GA * self.studying + B_inactive_CarAvail_GA * self.inactive + B_active_without_known_work_percentage_CarAvail_GA * self.active_without_known_work_percentage + B_no_post_school_educ_CarAvail_GA * self.no_post_school_educ + B_secundary_education_CarAvail_GA * self.secundary_education + B_tertiary_education_CarAvail_GA * self.tertiary_education + B_university_CarAvail_GA * self.university
        __CarAvail_HT = ASC_CarAvail_HT * self.one + B_MALE_CarAvail_HT * self.male + B_AGE_CarAvail_HT * self.age + B_AGE_SQUARE_CarAvail_HT * self.age_square_scaled + B_age_18_20_CarAvail_HT * self.age_18_20 + B_age_20_25_CarAvail_HT * self.age_20_25 + B_age_25_65_CarAvail_HT * self.age_25_65 + B_age_65_and_more_CarAvail_HT * self.age_65_and_more + B_EMPLOYED_CarAvail_HT * self.employed + B_AGE_TIME_MALE_CarAvail_HT * self.age_time_male_scaled + B_HH_INCOME_NA_CarAvail_HT * self.hh_income_na + B_HH_INCOME_LESS_THAN_4000_CarAvail_HT * self.hh_income_less_than_4000 + B_HH_INCOME_4001_to_10000_CarAvail_HT * self.hh_income_4001_to_10000 + B_HH_INCOME_MORE_THAN_10000_CarAvail_HT * self.hh_income_more_than_10000 + B_REGION_LAKE_GENEVA_CarAvail_HT * self.region_lake_geneva + B_REGION_ESPACE_MITTELLAND_CarAvail_HT * self.region_espace_mittelland + B_REGION_NORTHERN_SWITZERLAND_CarAvail_HT * self.region_northern_switzerland + B_REGION_ZURICH_CarAvail_HT * self.region_zurich + B_REGION_EASTERN_SWITZERLAND_CarAvail_HT * self.region_eastern_switzerland + B_REGION_CENTRAL_SWITZERLAND_CarAvail_HT * self.region_central_switzerland + B_REGION_TESSIN_CarAvail_HT * self.region_tessin + B_INHABITANTS_CarAvail_HT * self.pop_valid + B_LOG_INHABITANTS_CarAvail_HT * self.log_pop_valid + B_PT_QUALITY_A_CarAvail_HT * self.public_transport_connection_quality_ARE_A + B_PT_QUALITY_B_CarAvail_HT * self.public_transport_connection_quality_ARE_B + B_PT_QUALITY_C_CarAvail_HT * self.public_transport_connection_quality_ARE_C + B_PT_QUALITY_D_CarAvail_HT * self.public_transport_connection_quality_ARE_C + B_PT_QUALITY_NA_CarAvail_HT * self.public_transport_connection_quality_ARE_NA + B_single_household_CarAvail_HT * self.single_household + B_couple_without_children_CarAvail_HT * self.couple_without_children + B_couple_with_children_CarAvail_HT * self.couple_with_children + B_single_parent_with_children_CarAvail_HT * self.single_parent_with_children + B_adults_with_elderly_care_CarAvail_HT * self.adults_with_elderly_care + B_not_family_household_CarAvail_HT * self.not_family_household + B_full_time_work_CarAvail_HT * self.full_time_work + B_part_time_work_CarAvail_HT * self.part_time_work + B_studying_CarAvail_HT * self.studying + B_inactive_CarAvail_HT * self.inactive + B_active_without_known_work_percentage_CarAvail_HT * self.active_without_known_work_percentage + B_no_post_school_educ_CarAvail_HT * self.no_post_school_educ + B_secundary_education_CarAvail_HT * self.secundary_education + B_tertiary_education_CarAvail_HT * self.tertiary_education + B_university_CarAvail_HT * self.university
        __CarAvail_HT_Verbund = ASC_CarAvail_HT_Verbund * self.one + B_MALE_CarAvail_HT_Verbund * self.male + B_AGE_CarAvail_HT_Verbund * self.age + B_AGE_SQUARE_CarAvail_HT_Verbund * self.age_square_scaled + B_age_18_25_CarAvail_HT_Verbund * self.age_18_25 + B_age_25_65_CarAvail_HT_Verbund * self.age_25_65 + B_age_65_and_more_CarAvail_HT_Verbund * self.age_65_and_more + B_EMPLOYED_CarAvail_HT_Verbund * self.employed + B_AGE_TIME_MALE_CarAvail_HT_Verbund * self.age_time_male_scaled + B_HH_INCOME_NA_CarAvail_HT_Verbund * self.hh_income_na + B_HH_INCOME_LESS_THAN_4000_CarAvail_HT_Verbund * self.hh_income_less_than_4000 + B_HH_INCOME_4001_to_10000_CarAvail_HT_Verbund * self.hh_income_4001_to_10000 + B_HH_INCOME_MORE_THAN_10000_CarAvail_HT_Verbund * self.hh_income_more_than_10000 + B_REGION_LAKE_GENEVA_CarAvail_HT_Verbund * self.region_lake_geneva + B_REGION_ESPACE_MITTELLAND_CarAvail_HT_Verbund * self.region_espace_mittelland + B_REGION_NORTHERN_SWITZERLAND_CarAvail_HT_Verbund * self.region_northern_switzerland + B_REGION_ZURICH_CarAvail_HT_Verbund * self.region_zurich + B_REGION_EASTERN_SWITZERLAND_CarAvail_HT_Verbund * self.region_eastern_switzerland + B_REGION_CENTRAL_SWITZERLAND_CarAvail_HT_Verbund * self.region_central_switzerland + B_REGION_TESSIN_CarAvail_HT_Verbund * self.region_tessin + B_INHABITANTS_CarAvail_HT_Verbund * self.pop_valid + B_LOG_INHABITANTS_CarAvail_HT_Verbund * self.log_pop_valid + B_PT_QUALITY_A_CarAvail_HT_Verbund * self.public_transport_connection_quality_ARE_A + B_PT_QUALITY_B_CarAvail_HT_Verbund * self.public_transport_connection_quality_ARE_B + B_PT_QUALITY_C_CarAvail_HT_Verbund * self.public_transport_connection_quality_ARE_C + B_PT_QUALITY_D_CarAvail_HT_Verbund * self.public_transport_connection_quality_ARE_C + B_PT_QUALITY_NA_CarAvail_HT_Verbund * self.public_transport_connection_quality_ARE_NA + B_single_household_CarAvail_HT_Verbund * self.single_household + B_couple_without_children_CarAvail_HT_Verbund * self.couple_without_children + B_couple_with_children_CarAvail_HT_Verbund * self.couple_with_children + B_single_parent_with_children_CarAvail_HT_Verbund * self.single_parent_with_children + B_adults_with_elderly_care_CarAvail_HT_Verbund * self.adults_with_elderly_care + B_not_family_household_CarAvail_HT_Verbund * self.not_family_household + B_full_time_work_CarAvail_HT_Verbund * self.full_time_work + B_part_time_work_CarAvail_HT_Verbund * self.part_time_work + B_studying_CarAvail_HT_Verbund * self.studying + B_inactive_CarAvail_HT_Verbund * self.inactive + B_active_without_known_work_percentage_CarAvail_HT_Verbund * self.active_without_known_work_percentage + B_no_post_school_educ_CarAvail_HT_Verbund * self.no_post_school_educ + B_secundary_education_CarAvail_HT_Verbund * self.secundary_education + B_tertiary_education_CarAvail_HT_Verbund * self.tertiary_education + B_university_CarAvail_HT_Verbund * self.university
        __CarAvail_Verbund = ASC_CarAvail_Verbund * self.one + B_MALE_CarAvail_Verbund * self.male + B_AGE_CarAvail_Verbund * self.age + B_AGE_SQUARE_CarAvail_Verbund * self.age_square_scaled + B_age_18_25_CarAvail_Verbund * self.age_18_25 + B_age_25_65_CarAvail_Verbund * self.age_25_65 + B_age_65_and_more_CarAvail_Verbund * self.age_65_and_more + B_EMPLOYED_CarAvail_Verbund * self.employed + B_AGE_TIME_MALE_CarAvail_Verbund * self.age_time_male_scaled + B_HH_INCOME_NA_CarAvail_Verbund * self.hh_income_na + B_HH_INCOME_LESS_THAN_4000_CarAvail_Verbund * self.hh_income_less_than_4000 + B_HH_INCOME_4001_to_10000_CarAvail_Verbund * self.hh_income_4001_to_10000 + B_HH_INCOME_MORE_THAN_10000_CarAvail_Verbund * self.hh_income_more_than_10000 + B_REGION_LAKE_GENEVA_CarAvail_Verbund * self.region_lake_geneva + B_REGION_ESPACE_MITTELLAND_CarAvail_Verbund * self.region_espace_mittelland + B_REGION_NORTHERN_SWITZERLAND_CarAvail_Verbund * self.region_northern_switzerland + B_REGION_ZURICH_CarAvail_Verbund * self.region_zurich + B_REGION_EASTERN_SWITZERLAND_CarAvail_Verbund * self.region_eastern_switzerland + B_REGION_CENTRAL_SWITZERLAND_CarAvail_Verbund * self.region_central_switzerland + B_REGION_TESSIN_CarAvail_Verbund * self.region_tessin + B_INHABITANTS_CarAvail_Verbund * self.pop_valid + B_LOG_INHABITANTS_CarAvail_Verbund * self.log_pop_valid + B_PT_QUALITY_A_CarAvail_Verbund * self.public_transport_connection_quality_ARE_A + B_PT_QUALITY_B_CarAvail_Verbund * self.public_transport_connection_quality_ARE_B + B_PT_QUALITY_C_CarAvail_Verbund * self.public_transport_connection_quality_ARE_C + B_PT_QUALITY_D_CarAvail_Verbund * self.public_transport_connection_quality_ARE_C + B_PT_QUALITY_NA_CarAvail_Verbund * self.public_transport_connection_quality_ARE_NA + B_single_household_CarAvail_Verbund * self.single_household + B_couple_without_children_CarAvail_Verbund * self.couple_without_children + B_couple_with_children_CarAvail_Verbund * self.couple_with_children + B_single_parent_with_children_CarAvail_Verbund * self.single_parent_with_children + B_adults_with_elderly_care_CarAvail_Verbund * self.adults_with_elderly_care + B_not_family_household_CarAvail_Verbund * self.not_family_household + B_full_time_work_CarAvail_Verbund * self.full_time_work + B_part_time_work_CarAvail_Verbund * self.part_time_work + B_studying_CarAvail_Verbund * self.studying + B_inactive_CarAvail_Verbund * self.inactive + B_active_without_known_work_percentage_CarAvail_Verbund * self.active_without_known_work_percentage + B_no_post_school_educ_CarAvail_Verbund * self.no_post_school_educ + B_secundary_education_CarAvail_Verbund * self.secundary_education + B_tertiary_education_CarAvail_Verbund * self.tertiary_education + B_university_CarAvail_Verbund * self.university
        __GA = ASC_GA_scaled * self.ten + B_MALE_GA * self.male + B_AGE_GA * self.age + B_AGE_SQUARE_GA * self.age_square_scaled + B_age_6_16_GA * self.age_6_16 + B_age_16_18_GA * self.age_16_18 + B_age_18_20_GA * self.age_18_20 + B_age_20_45_GA * self.age_20_45 + B_age_45_65_GA * self.age_45_65 + B_age_65_and_more_GA * self.age_65_and_more + B_EMPLOYED_GA * self.employed + B_AGE_TIME_MALE_GA * self.age_time_male_scaled + B_HH_INCOME_NA_GA * self.hh_income_na + B_HH_INCOME_LESS_THAN_4000_GA * self.hh_income_less_than_4000 + B_HH_INCOME_4001_to_10000_GA * self.hh_income_4001_to_10000 + B_HH_INCOME_MORE_THAN_10000_GA * self.hh_income_more_than_10000 + B_REGION_LAKE_GENEVA_GA * self.region_lake_geneva + B_REGION_ESPACE_MITTELLAND_GA * self.region_espace_mittelland + B_REGION_NORTHERN_SWITZERLAND_GA * self.region_northern_switzerland + B_REGION_ZURICH_GA * self.region_zurich + B_REGION_EASTERN_SWITZERLAND_GA * self.region_eastern_switzerland + B_REGION_CENTRAL_SWITZERLAND_GA * self.region_central_switzerland + B_REGION_TESSIN_GA * self.region_tessin + B_INHABITANTS_GA * self.pop_valid + B_LOG_INHABITANTS_GA * self.log_pop_valid + B_PT_QUALITY_A_GA * self.public_transport_connection_quality_ARE_A + B_PT_QUALITY_B_GA * self.public_transport_connection_quality_ARE_B + B_PT_QUALITY_C_GA * self.public_transport_connection_quality_ARE_C + B_PT_QUALITY_D_GA * self.public_transport_connection_quality_ARE_C + B_PT_QUALITY_NA_GA * self.public_transport_connection_quality_ARE_NA + B_single_household_GA * self.single_household + B_couple_without_children_GA * self.couple_without_children + B_couple_with_children_GA * self.couple_with_children + B_single_parent_with_children_GA * self.single_parent_with_children + B_adults_with_elderly_care_GA * self.adults_with_elderly_care + B_not_family_household_GA * self.not_family_household + B_full_time_work_GA * self.full_time_work + B_part_time_work_GA * self.part_time_work + B_studying_GA * self.studying + B_inactive_GA * self.inactive + B_active_without_known_work_percentage_GA * self.active_without_known_work_percentage + B_no_post_school_educ_GA * self.no_post_school_educ + B_secundary_education_GA * self.secundary_education + B_tertiary_education_GA * self.tertiary_education + B_university_GA * self.university
        __HT = ASC_HT * self.one + B_MALE_HT * self.male + B_AGE_HT * self.age + B_AGE_SQUARE_HT * self.age_square_scaled + B_age_16_18_HT * self.age_6_18 + B_age_18_20_HT * self.age_18_20 + B_age_20_25_HT * self.age_20_25 + B_age_25_65_HT * self.age_25_65 + B_age_65_and_more_HT * self.age_65_and_more + B_EMPLOYED_HT * self.employed + B_AGE_TIME_MALE_HT * self.age_time_male_scaled + B_HH_INCOME_NA_HT * self.hh_income_na + B_HH_INCOME_LESS_THAN_4000_HT * self.hh_income_less_than_4000 + B_HH_INCOME_4001_to_10000_HT * self.hh_income_4001_to_10000 + B_HH_INCOME_MORE_THAN_10000_HT * self.hh_income_more_than_10000 + B_REGION_LAKE_GENEVA_HT * self.region_lake_geneva + B_REGION_ESPACE_MITTELLAND_HT * self.region_espace_mittelland + B_REGION_NORTHERN_SWITZERLAND_HT * self.region_northern_switzerland + B_REGION_ZURICH_HT * self.region_zurich + B_REGION_EASTERN_SWITZERLAND_HT * self.region_eastern_switzerland + B_REGION_CENTRAL_SWITZERLAND_HT * self.region_central_switzerland + B_REGION_TESSIN_HT * self.region_tessin + B_INHABITANTS_HT * self.pop_valid + B_LOG_INHABITANTS_HT * self.log_pop_valid + B_PT_QUALITY_A_HT * self.public_transport_connection_quality_ARE_A + B_PT_QUALITY_B_HT * self.public_transport_connection_quality_ARE_B + B_PT_QUALITY_C_HT * self.public_transport_connection_quality_ARE_C + B_PT_QUALITY_D_HT * self.public_transport_connection_quality_ARE_C + B_PT_QUALITY_NA_HT * self.public_transport_connection_quality_ARE_NA + B_single_household_HT * self.single_household + B_couple_without_children_HT * self.couple_without_children + B_couple_with_children_HT * self.couple_with_children + B_single_parent_with_children_HT * self.single_parent_with_children + B_adults_with_elderly_care_HT * self.adults_with_elderly_care + B_not_family_household_HT * self.not_family_household + B_full_time_work_HT * self.full_time_work + B_part_time_work_HT * self.part_time_work + B_studying_HT * self.studying + B_inactive_HT * self.inactive + B_active_without_known_work_percentage_HT * self.active_without_known_work_percentage + B_no_post_school_educ_HT * self.no_post_school_educ + B_secundary_education_HT * self.secundary_education + B_tertiary_education_HT * self.tertiary_education + B_university_HT * self.university
        __HT_Verbund = ASC_HT_Verbund * self.one + B_MALE_HT_Verbund * self.male + B_AGE_HT_Verbund * self.age + B_AGE_SQUARE_HT_Verbund * self.age_square_scaled + B_age_16_18_HT_Verbund * self.age_6_18 + B_age_18_20_HT_Verbund * self.age_18_20 + B_age_20_25_HT_Verbund * self.age_20_25 + B_age_25_45_HT_Verbund * self.age_25_45 + B_age_45_65_HT_Verbund * self.age_45_65 + B_age_65_and_more_HT_Verbund * self.age_65_and_more + B_EMPLOYED_HT_Verbund * self.employed + B_AGE_TIME_MALE_HT_Verbund * self.age_time_male_scaled + B_HH_INCOME_NA_HT_Verbund * self.hh_income_na + B_HH_INCOME_LESS_THAN_4000_HT_Verbund * self.hh_income_less_than_4000 + B_HH_INCOME_4001_to_10000_HT_Verbund * self.hh_income_4001_to_10000 + B_HH_INCOME_MORE_THAN_10000_HT_Verbund * self.hh_income_more_than_10000 + B_REGION_LAKE_GENEVA_HT_Verbund * self.region_lake_geneva + B_REGION_ESPACE_MITTELLAND_HT_Verbund * self.region_espace_mittelland + B_REGION_NORTHERN_SWITZERLAND_HT_Verbund * self.region_northern_switzerland + B_REGION_ZURICH_HT_Verbund * self.region_zurich + B_REGION_EASTERN_SWITZERLAND_HT_Verbund * self.region_eastern_switzerland + B_REGION_CENTRAL_SWITZERLAND_HT_Verbund * self.region_central_switzerland + B_REGION_TESSIN_HT_Verbund * self.region_tessin + B_INHABITANTS_HT_Verbund * self.pop_valid + B_LOG_INHABITANTS_HT_Verbund * self.log_pop_valid + B_PT_QUALITY_A_HT_Verbund * self.public_transport_connection_quality_ARE_A + B_PT_QUALITY_B_HT_Verbund * self.public_transport_connection_quality_ARE_B + B_PT_QUALITY_C_HT_Verbund * self.public_transport_connection_quality_ARE_C + B_PT_QUALITY_D_HT_Verbund * self.public_transport_connection_quality_ARE_C + B_PT_QUALITY_NA_HT_Verbund * self.public_transport_connection_quality_ARE_NA + B_single_household_HT_Verbund * self.single_household + B_couple_without_children_HT_Verbund * self.couple_without_children + B_couple_with_children_HT_Verbund * self.couple_with_children + B_single_parent_with_children_HT_Verbund * self.single_parent_with_children + B_adults_with_elderly_care_HT_Verbund * self.adults_with_elderly_care + B_not_family_household_HT_Verbund * self.not_family_household + B_full_time_work_HT_Verbund * self.full_time_work + B_part_time_work_HT_Verbund * self.part_time_work + B_studying_HT_Verbund * self.studying + B_inactive_HT_Verbund * self.inactive + B_active_without_known_work_percentage_HT_Verbund * self.active_without_known_work_percentage + B_no_post_school_educ_HT_Verbund * self.no_post_school_educ + B_secundary_education_HT_Verbund * self.secundary_education + B_tertiary_education_HT_Verbund * self.tertiary_education + B_university_HT_Verbund * self.university
        __No_Mobility_Tool = ASC_No_Mobility_Tool * self.one
        __Verbund = ASC_Verbund * self.one + B_MALE_Verbund * self.male + B_AGE_Verbund * self.age + B_AGE_SQUARE_Verbund * self.age_square_scaled + B_age_6_16_Verbund * self.age_6_16 + B_age_16_18_Verbund * self.age_16_18 + B_age_18_20_Verbund * self.age_18_20 + B_age_20_25_Verbund * self.age_20_25 + B_age_25_45_Verbund * self.age_25_45 + B_age_45_65_Verbund * self.age_45_65 + B_age_65_and_more_Verbund * self.age_65_and_more + B_EMPLOYED_Verbund * self.employed + B_AGE_TIME_MALE_Verbund * self.age_time_male_scaled + B_HH_INCOME_NA_Verbund * self.hh_income_na + B_HH_INCOME_LESS_THAN_4000_Verbund * self.hh_income_less_than_4000 + B_HH_INCOME_4001_to_10000_Verbund * self.hh_income_4001_to_10000 + B_HH_INCOME_MORE_THAN_10000_Verbund * self.hh_income_more_than_10000 + B_REGION_LAKE_GENEVA_Verbund * self.region_lake_geneva + B_REGION_ESPACE_MITTELLAND_Verbund * self.region_espace_mittelland + B_REGION_NORTHERN_SWITZERLAND_Verbund * self.region_northern_switzerland + B_REGION_ZURICH_Verbund * self.region_zurich + B_REGION_EASTERN_SWITZERLAND_Verbund * self.region_eastern_switzerland + B_REGION_CENTRAL_SWITZERLAND_Verbund * self.region_central_switzerland + B_REGION_TESSIN_Verbund * self.region_tessin + B_INHABITANTS_Verbund * self.pop_valid + B_LOG_INHABITANTS_Verbund * self.log_pop_valid + B_PT_QUALITY_A_Verbund * self.public_transport_connection_quality_ARE_A + B_PT_QUALITY_B_Verbund * self.public_transport_connection_quality_ARE_B + B_PT_QUALITY_C_Verbund * self.public_transport_connection_quality_ARE_C + B_PT_QUALITY_D_Verbund * self.public_transport_connection_quality_ARE_C + B_PT_QUALITY_NA_Verbund * self.public_transport_connection_quality_ARE_NA + B_single_household_Verbund * self.single_household + B_couple_without_children_Verbund * self.couple_without_children + B_couple_with_children_Verbund * self.couple_with_children + B_single_parent_with_children_Verbund * self.single_parent_with_children + B_adults_with_elderly_care_Verbund * self.adults_with_elderly_care + B_not_family_household_Verbund * self.not_family_household + B_full_time_work_Verbund * self.full_time_work + B_part_time_work_Verbund * self.part_time_work + B_studying_Verbund * self.studying + B_inactive_Verbund * self.inactive + B_active_without_known_work_percentage_Verbund * self.active_without_known_work_percentage + B_no_post_school_educ_Verbund * self.no_post_school_educ + B_secundary_education_Verbund * self.secundary_education + B_tertiary_education_Verbund * self.tertiary_education + B_university_Verbund * self.university


        # Associate utility functions with the numbering of alternatives
        self.__V = {3: __CarAvail, 1: __CarAvail_GA, 2: __CarAvail_HT, 20: __CarAvail_HT_Verbund, 30: __CarAvail_Verbund,
               4: __GA, 5: __HT, 50: __HT_Verbund, 6: __No_Mobility_Tool, 60: __Verbund}

        # Associate the availability conditions with the alternatives
        self.__av = {3: self.avail_car, 1: self.avail_car, 2: self.avail_car, 20: self.avail_car, 30: self.avail_car, 4: self.one, 5: self.avail_HT,
                50: self.avail_HT, 6: self.one, 60: self.one}

        logprob = bioLogLogit(self.__V, self.__av, self.choice)
        self.biogeme = bio.BIOGEME(self.database, logprob)
        self.biogeme.modelName = "MTMC_MNL"
        self.biogeme.generateHtml = False

        self.x0 = self.biogeme.betaInitValues

    def sample(self, n):
        if n < self.size_db:
            sample = self.biogeme.database.sample(n)
            self.biogeme.theC.setData(sample)
        else:
            self.biogeme.theC.setData(self.database.data)

    def optimize(self, algo, **kwargs):

        self.biogeme.database = self.database
        self.biogeme.theC.setData(self.biogeme.database.data)

        algo.__prep__(self.x0, self.biogeme, **kwargs)

        return algo.solve(maximize=True)

    def simulate(self):
        prob_CarAvail = bioLogLogit(self.__V, self.__av, 3)
        prob_CarAvail_GA = bioLogLogit(self.__V, self.__av, 1)
        prob_CarAvail_HT = bioLogLogit(self.__V, self.__av, 2)
        prob_CarAvail_HT_Verbund = bioLogLogit(self.__V, self.__av, 20)
        prob_CarAvail_Verbund = bioLogLogit(self.__V, self.__av, 30)
        prob_GA = bioLogLogit(self.__V, self.__av, 4)
        prob_HT = bioLogLogit(self.__V, self.__av, 5)
        prob_HT_Verbund = bioLogLogit(self.__V, self.__av, 50)
        prob_No_Mobility_Tool = bioLogLogit(self.__V, self.__av, 6)
        prob_Verbund = bioLogLogit(self.__V, self.__av, 60)

        simulate = {'prob_CarAvail': prob_CarAvail,
                    'prob_CarAvail_GA': prob_CarAvail_GA,
                    'prob_CarAvail_HT': prob_CarAvail_HT,
                    'prob_CarAvail_HT_Verbund': prob_CarAvail_HT_Verbund,
                    'prob_CarAvail_Verbund': prob_CarAvail_Verbund,
                    'prob_GA': prob_GA,
                    'prob_HT': prob_HT,
                    'prob_HT_Verbund': prob_HT_Verbund,
                    'prob_No_Mobility_Tool': prob_No_Mobility_Tool,
                    'prob_Verbund': prob_Verbund}

        self.biogeme = bio.BIOGEME(self.database, simulate)

        return self.biogeme.simulate()

