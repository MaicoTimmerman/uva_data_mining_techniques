import pandas as pd
from datetime import datetime
from datetime import timedelta
import numpy as np
from visualization import scatterplotter, correlation_matrixo


def load_the_datas(filename='tiny_train.csv'):
    types = {'srch_id': int,
             'site_id': int,
             'click_bool': bool,
             'booking_bool': bool,
             'gross_booking_uds': float,
             'visitor_location_country_id': int,
             'visitor_hist_starrating': float,
             'visitor_hist_adr_usd': float,
             'prop_country_id': int,
             'prop_id': int,
             'prop_starrating': int,
             'prop_review_score': float,
             'prop_brand_bool': int,
             'prop_location_score1': float,
             'prop_location_score2': float,
             'prop_log_historical_price': float,
             'price_usd': float,
             'promotion_flag': int,
             'srch_destination_id': int,
             'srch_length_of_stay': int,
             'srch_booking_window': int,
             'srch_adults_count': int,
             'srch_children_count': int,
             'srch_room_count': int,
             'srch_saturday_night_bool': bool,
             'srch_query_affinity_score': float,
             'orig_destination_distance': float,
             'random_bool': bool,
             'comp1_rate' : 'Int64',
             'comp1_inv': 'Int64',
             'comp1_rate_percent_diff': float,
             'comp2_rate': 'Int64',
             'comp2_inv': 'Int64',
             'comp2_rate_percent_diff': float,
             'comp3_rate': 'Int64',
             'comp3_inv': 'Int64',
             'comp3_rate_percent_diff': float,
             'comp4_rate': 'Int64',
             'comp4_inv': 'Int64',
             'comp4_rate_percent_diff': float,
             'comp5_rate': 'Int64',
             'comp5_inv': 'Int64',
             'comp5_rate_percent_diff': float,
             'comp6_rate': 'Int64',
             'comp6_inv': 'Int64',
             'comp6_rate_percent_diff': float,
             'comp7_rate': 'Int64',
             'comp7_inv': 'Int64',
             'comp7_rate_percent_diff': float,
             'comp8_rate': 'Int64',
             'comp8_inv': 'Int64',
             'comp8_rate_percent_diff': float}

    parse_dates = ['date_time']
    df = pd.read_csv(filename, dtype=types, index_col=['srch_id', 'position', 'prop_id', 'click_bool',
    'booking_bool'], parse_dates=parse_dates)

    df.sort_index(level=['srch_id', 'position'], inplace=True)

    return df

def add_seasons(df):

    df['srch_booking_window'] = df['srch_booking_window'].apply(lambda x: timedelta(days = x))
    df['time_of_booking'] = df['date_time'] + df['srch_booking_window']

    df['SPRING'] = np.where(pd.DatetimeIndex(df.time_of_booking).month.isin([3,4,5]), 1, 0)
    df['SUMMER'] = np.where(pd.DatetimeIndex(df.time_of_booking).month.isin([6,7,8]), 1, 0)
    df['AUTUMN'] = np.where(pd.DatetimeIndex(df.time_of_booking).month.isin([9,10,11]), 1, 0)
    df['WINTER'] = np.where(pd.DatetimeIndex(df.time_of_booking).month.isin([12,1,2]), 1, 0)

    return df

def average_competitors(df):
    all_comp_rates = ['comp' + str(x) + '_rate' for x in range(1,9)]
    all_comp_invs = ['comp' + str(x) + '_inv' for x in range(1,9)]
    all_comp_diff = ['comp' + str(x) + '_rate_percent_diff' for x in range(1,9)]

    df['comp_rate_avg'] = df[all_comp_rates].mean(axis=1)
    df['comp_inv_avg'] = df[all_comp_invs].mean(axis=1)
    df['comp_diff_avg'] = df[all_comp_diff].mean(axis=1)

    df.drop(all_comp_rates + all_comp_invs + all_comp_diff, axis=1).head()
    return df



if __name__ == "__main__":

    path = '../data/tiny_train.csv'

    df = load_the_datas(path)
    df = add_seasons(df)
    df = average_competitors(df)

    scatterplotter(df)
    # correlation_matrixo(df)
