import pandas as pd
from datetime import datetime
from datetime import timedelta
import numpy as np
from sklearn import cluster

from visualization import   scatterplotter, \
                            correlation_matrixo, \
                            show_country_clusters, \
                            showNaNs, \
                            box_plot_variable, \
                            show_me_the_money, \
                            do_you_like_pie

def load_the_datas(filename='tiny_train.csv'):
    types = {'srch_id': int,
             'site_id': int,
             'click_bool': int,
             'booking_bool': int,
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
             'srch_saturday_night_bool': int,
             'srch_query_affinity_score': float,
             'orig_destination_distance': float,
             'random_bool': int,
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
    df = pd.read_csv(filename, dtype=types, index_col=['srch_id', 'position'], parse_dates=parse_dates)

    df.sort_index(level=['srch_id', 'position'], inplace=True)

    return df

def add_seasons(df):
    df['time_of_check_in'] = df['date_time'] + df['srch_booking_window'].apply(lambda x: timedelta(x))
    df['time_of_check_out'] = df['time_of_check_in'] + df['srch_length_of_stay'].apply(lambda x: timedelta(x))
    check_ins = df['time_of_check_in'].apply(lambda x: x.timetuple().tm_yday) / 365
    df['sin_day_of_year_check_in'] = np.sin(2*np.pi*check_ins)
    df['cos_day_of_year_check_in'] = np.cos(2*np.pi*check_ins)

    df['SPRING'] = np.where(pd.DatetimeIndex(df.time_of_check_in).month.isin([3,4,5]), 1, 0)
    df['SUMMER'] = np.where(pd.DatetimeIndex(df.time_of_check_in).month.isin([6,7,8]), 1, 0)
    df['AUTUMN'] = np.where(pd.DatetimeIndex(df.time_of_check_in).month.isin([9,10,11]), 1, 0)
    df['WINTER'] = np.where(pd.DatetimeIndex(df.time_of_check_in).month.isin([12,1,2]), 1, 0)

    return df

def average_competitors(df):
    all_comp_rates = ['comp' + str(x) + '_rate' for x in range(1,9)]
    all_comp_invs = ['comp' + str(x) + '_inv' for x in range(1,9)]
    all_comp_diff = ['comp' + str(x) + '_rate_percent_diff' for x in range(1,9)]

    df['comp_rate_avg'] = df[all_comp_rates].mean(axis=1)
    df['comp_inv_avg'] = df[all_comp_invs].mean(axis=1)
    df['comp_diff_avg'] = df[all_comp_diff].mean(axis=1)

    df.drop(all_comp_rates + all_comp_invs + all_comp_diff, axis=1, inplace=True)
    return df

def remove_nans(df):
    """
        removal of NaNs has to be different per column. For many we can simply
        replace with -1. But for affinity score we set to 0 as all values are negative
        for competitors we set to 0, because yea, no competition no problem
    """

    values = {  'visitor_hist_starrating': -1,
                'visitor_hist_adr_usd': -1,
                'prop_review_score': -1,
                'prop_location_score2': -1,
                'srch_query_affinity_score': 0,
                'orig_destination_distance': -1,
                'gross_bookings_usd': -1,
                'comp_rate_avg': 0,
                'comp_inv_avg': 0,
                'comp_diff_avg': 0,}
    df.fillna(value=values,inplace=True)
    return df

def cluster_hotel_countries(df, k=4):
    """
        We cluster the hotels' countries based on the mean and standard distribution of
        each countries hotel prices.
        Furthermore we add country_mean_price and country_std_price as new features

        TODO: clusters are now valued between 0-k, but these should become one-hot
        vectors.
    """
    # Convert DataFrame to matrix
    mat = df.groupby("prop_country_id").price_usd.describe()[['mean','std']]
    # Using sklearn
    km = cluster.KMeans(n_clusters=k)
    km.fit(mat.values)
    # Get cluster assignment labels
    labels = km.labels_
    # Format results as a DataFrame
    results = pd.DataFrame([df.groupby("prop_country_id").price_usd.describe()[['mean','std']].index,labels]).T

    df["country_cluster"] = results.set_index(0).loc[df.prop_country_id].values

    for one_hot_index in range(df["country_cluster"].max() + 1):
        variable_name = "hotel_country_cluster_" + str(one_hot_index)
        df[variable_name] = (df["country_cluster"] == one_hot_index).astype(int)

    df["country_mean_price"] = mat.loc[df.prop_country_id]['mean'].values
    df["country_std_price"] = mat.loc[df.prop_country_id]['std'].values

    return df

def cluster_user_countries(df):
    """
        I want to cluster the users by their countries. I want to use a similar
        approach to clustering hotels by country. Perhaps I want to aggregate
        all users by country, and plot their spending.
    """
    mat = df.loc[df['booking_bool'] == 1].groupby("visitor_location_country_id").price_usd.describe()[['mean','std']]

    # print(mat, len(mat))
    # print(df.visitor_location_country_id.unique(), len(df.visitor_location_country_id.unique()))

    df["user_country_mean_spending"] = mat.loc[df.visitor_location_country_id]['mean'].values
    df["user_country_std_spending"] = mat.loc[df.visitor_location_country_id]['std'].values
    df.fillna({'user_country_mean_spending': 0, 'user_country_std_spending': 0}, inplace=True)
    return df

def create_cheats_sheet(df):
    """
        We should only use non-shuffled data. Otherwise we would just be adding noise.
    """
    temp_df = df.reset_index()
    # print(temp_df.shape)
    temp_df = temp_df.loc[temp_df['random_bool'] == 0]
    # print(temp_df.shape)

    assert 'position' in temp_df.columns, ("Please use training data to create cheat sheet")
    assert 'booking_bool' in temp_df.columns, ("Please use training data to create cheat sheet")
    # assert 'gross_booking_usd' in temp_df.columns, ("Please use training data to create cheat sheet")
    assert 'click_bool' in temp_df.columns, ("Please use training data to create cheat sheet")



    booker_bools = temp_df.groupby('prop_id')['booking_bool'].sum()/temp_df.groupby('prop_id')['booking_bool'].count()
    clicker_bools = temp_df.groupby('prop_id')['click_bool'].sum()/temp_df.groupby('prop_id')['click_bool'].count()

    cheat_cheet = temp_df.groupby('prop_id')['position'].describe()[['mean','std']]
    cheat_cheet.fillna({'std': 0}, inplace=True)

    cheat_cheet['click_ratio'] = clicker_bools
    cheat_cheet['book_ratio'] = booker_bools

    return cheat_cheet

def property_id_hacking(df, cheat_sheet):

    df["hotel_position_mean"] = cheat_sheet.loc[df.prop_id]['mean'].values
    df["hotel_position_std"] = cheat_sheet.loc[df.prop_id]['std'].values
    df["hotel_clicked_ratio"] = cheat_sheet.loc[df.prop_id]['click_ratio'].values
    df["hotel_booked_ratio"] = cheat_sheet.loc[df.prop_id]['book_ratio'].values

    df.fillna({ 'hotel_position_mean': 40,
                'hotel_position_std': 0,
                'hotel_clicked_ratio': 0,
                'hotel_booked_ratio': 0,}, inplace=True)
    return df

if __name__ == "__main__":

    training = True
    no_cheat_sheet = True

    path = '../data/tiny_train.csv'

    df = load_the_datas(path)
    df = add_seasons(df)
    df = average_competitors(df)
    df = remove_nans(df)
    df = cluster_hotel_countries(df)
    df = cluster_user_countries(df)

    if training and no_cheat_sheet:
        cheat_sheet = create_cheats_sheet(df)

    df = property_id_hacking(df, cheat_sheet)

    # showNaNs(df)
    box_plot_variable(df)
    show_me_the_money(df)
    # do_you_like_pie(df)
    # VISUALS
    # scatterplotter(df)
    # correlation_matrixo(df)
    # show_country_clusters(df)

    print("done")
