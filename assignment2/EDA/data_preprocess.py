import pickle
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn import cluster
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import argparse
from pathlib import Path

from visualization import   scatterplotter, \
                            correlation_matrixo, \
                            show_country_clusters, \
                            showNaNs, \
                            box_plot_variable, \
                            show_me_the_money, \
                            do_you_like_pie, \
                            polar_graph, \
                            country_price_vagina_plots, \
                            position_bias, \
                            histogram_site_id, \
                            booking_days, \
                            polar_booking_days
"""XGboost?"""


def load_the_datas(filename, set_type):
    types = {'srch_id': int,
             'site_id': int,
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

    if set_type == "training" or set_type == "validation":
        types.update({
            'click_bool': int,
            'booking_bool': int,
            'gross_bookings_usd': float,
        })
        index_cols = ['srch_id', 'position']
    else:
        index_cols = ['srch_id']
    parse_dates = ['date_time']
    df = pd.read_csv(filename, dtype=types, index_col=index_cols,
                     parse_dates=parse_dates)

    are_training_columns_present = "position" in df.columns
    if set_type == "test" and are_training_columns_present:
        df.drop(columns=["position", "booking_bool", "click_bool", "gross_bookings_usd"], inplace=True)

    if set_type == "training" or set_type == "validation":
        df.sort_index(level=['srch_id', 'position'], inplace=True)

    return df

def add_circular_day_of_year_features (df):
    pd.options.mode.chained_assignment = None  # default='warn'
    df['time_of_check_in'] = df['date_time'] + df['srch_booking_window'].apply(lambda x: timedelta(x))
    df['time_of_check_out'] = df['time_of_check_in'] + df['srch_length_of_stay'].apply(lambda x: timedelta(x))
    booking_doys = df['date_time'].apply(lambda x: x.timetuple().tm_yday) / 365
    check_in_doys = df['time_of_check_in'].apply(lambda x: x.timetuple().tm_yday) / 365
    check_out_doys = df['time_of_check_out'].apply(lambda x: x.timetuple().tm_yday) / 365
    df['sin_day_of_year_booking'] = np.sin(2*np.pi*booking_doys)
    df['cos_day_of_year_booking'] = np.cos(2*np.pi*booking_doys)
    df['sin_day_of_year_check_in'] = np.sin(2*np.pi*check_in_doys)
    df['cos_day_of_year_check_in'] = np.cos(2*np.pi*check_in_doys)
    df['sin_day_of_year_check_out'] = np.sin(2*np.pi*check_out_doys)
    df['cos_day_of_year_check_out'] = np.cos(2*np.pi*check_out_doys)
    pd.options.mode.chained_assignment = 'warn'
    return df

def add_seasons(df):
    pd.options.mode.chained_assignment = None  # default='warn'
    df['WEEKEND_date_time'] = ((pd.DatetimeIndex(df.date_time).dayofweek) // 5 == 1).astype(int)
    df['WEEKEND_check_in'] = ((pd.DatetimeIndex(df.time_of_check_in).dayofweek) // 5 == 1).astype(int)
    df['SPRING'] = np.where(pd.DatetimeIndex(df.time_of_check_in).month.isin([3,4,5]), 1, 0)
    df['SUMMER'] = np.where(pd.DatetimeIndex(df.time_of_check_in).month.isin([6,7,8]), 1, 0)
    df['AUTUMN'] = np.where(pd.DatetimeIndex(df.time_of_check_in).month.isin([9,10,11]), 1, 0)
    df['WINTER'] = np.where(pd.DatetimeIndex(df.time_of_check_in).month.isin([12,1,2]), 1, 0)
    pd.options.mode.chained_assignment = 'warn'
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

    print(f"{df.isnull().values.any(axis=1).sum()} rows with NaNs in dataset after competitor averaging to be filled.")
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
        Furthermore we add hotel_country_mean_price and country_std_price as new features

    """
    # Convert DataFrame to matrix
    mat = df.groupby("prop_country_id").price_usd.describe()[['mean','std']]
    mat.rename(columns = {"mean": 'prop_country_id_price_usd_mean', 'std': 'prop_country_id_price_usd_std'}, inplace=True)
    # Using sklearn
    km = cluster.KMeans(n_clusters=k)
    # print(mat.isna().sum().sort_values())
    mat = mat.fillna(0)
    km.fit(mat.values)
    # Get cluster assignment labels
    labels = km.labels_
    # Format results as a DataFrame
    results = pd.DataFrame([df.groupby("prop_country_id").price_usd.describe()[['mean','std']].index,labels]).T

    pd.options.mode.chained_assignment = None  # default='warn'
    df["country_cluster"] = results.set_index(0).loc[df.prop_country_id].values

    for one_hot_index in range(df["country_cluster"].max() + 1):
        variable_name = "hotel_country_cluster_" + str(one_hot_index)
        df[variable_name] = (df["country_cluster"] == one_hot_index).astype(int)

    pd.options.mode.chained_assignment = 'warn'
    return df

def id_hacking(df):
    """
        First make the cheat sheets with the create_cheat_sheet
        Then you can load in the data
    """
    def fill_with_cheats(df, pickle_file):


        file_stream = open(pickle_file, 'rb')
        cheat_sheet = pickle.load(file_stream)

        df = df.merge(cheat_sheet, how='left', left_on=cheat_sheet.index.name, right_on=cheat_sheet.index.name)

        for variable_name in cheat_sheet.columns:
            df.fillna({variable_name: 0}, inplace=True)
        print("finished merging {} with dataframe".format(pickle_file))
        return df

    file_stream = open('../data/cheat_sheet_prop_id.pkl', 'rb')
    cheat_sheet = pickle.load(file_stream)

    df = df.reset_index()
    df = df.merge(cheat_sheet, how='left', left_on=cheat_sheet.index.name, right_on=cheat_sheet.index.name)

    print(f"Prop_id cheat sheet left {df.isnull().values.any(axis=1).sum()} rows with NaNs to be removed")
    df.fillna({ 'prop_id_position_mean': 40, 'prop_id_position_std': 0,
                'prop_id_clicked_ratio': 0, 'prop_id_book_ratio': 0,}, inplace=True)

    df = fill_with_cheats(df, '../data/cheat_sheet_visitor_location_country_id.pkl')
    df = fill_with_cheats(df, '../data/cheat_sheet_site_id.pkl')
    df = fill_with_cheats(df, '../data/cheat_sheet_srch_destination_id.pkl')
    df = fill_with_cheats(df, '../data/cheat_sheet_prop_country_id.pkl')

    # df.set_index(['srch_id', 'position'], inplace=True)

    return df

def outlier_killer(df):
    q = df["price_usd"].quantile(0.999)
    # q = 4000
    # return df[(df["price_usd"] < q) & (df["booking_bool"] == 1)]
    return df[(df["price_usd"] < q)]

def balance_relevancies_stupid (df):
    df_copy = pd.DataFrame().reindex_like(df)
    for srch_id, df2 in df.groupby('srch_id'):
        for i, row in df2.iterrows():
            if row['click_bool'] == 1 & row['booking_bool'] == 1:
                df_copy[-1] = row
                break

        for i, row in df2.iterrows():
            if row['click_bool'] == 1 & row['booking_bool'] == 0:
                df_copy[-1] = row
                break

        for i, row in df2.iterrows():
            if row['click_bool'] == 0 & row['booking_bool'] == 0:
                df_copy[-1] = row
                break

    return df_copy

def balance_relevancies (df):
    # df.assign(srch_id2=df.index.get_level_values('srch_id'))
    df = df.reset_index()
    df = df.drop_duplicates(subset=['srch_id', 'relevance'])
    df = df.set_index(['srch_id', 'position'])
    return df


def balance_sampling(df, n=10):
    print(f"Take top {n} positions and add clicked samples from {df.shape} dataframe.")
    # for srch_id, df2 in df.groupby('srch_id'):
    #
    #     df.drop(index=df2[df2['relevance'] == 0][5:].index.values,
    #             axis=1, inplace=True)

    df.reset_index(inplace=True)
    df_new = df.groupby("srch_id").position.nsmallest(n, keep='first')
    df_new = df_new.reset_index().drop("level_1", axis=1)

    df_new.set_index(['srch_id', 'position'], inplace=True)
    df_new.sort_index(level=['srch_id', 'position'], inplace=True)

    df.set_index(['srch_id', 'position'], inplace=True)
    df.sort_index(level=['srch_id', 'position'], inplace=True)

    df_new = df_new.join(df)
    df_new = df_new.combine_first(df[df['click_bool'] == 1])
    df_new = df_new.reset_index().drop("index", axis=1)
    print(f"New dataframe has size {df_new.shape}.")
    return df_new

def normalizer(df, normalize=True):

    """
        Normalization of data causes the visualizations to often bug or give misleading stats
        be careful when combining normalization and visualizations!
    """


    things_to_normalize = [ 'visitor_hist_starrating', 'visitor_hist_adr_usd',
                            'prop_starrating', 'prop_review_score', 'prop_location_score1',
                            'prop_location_score2', 'prop_log_historical_price',
                            'price_usd', 'srch_length_of_stay', 'srch_booking_window',
                            'srch_adults_count', 'srch_children_count', 'srch_room_count',
                            'srch_query_affinity_score', 'orig_destination_distance',
                            'comp_rate_avg', 'comp_inv_avg', 'comp_diff_avg',
                            'hotel_country_mean_price', 'hotel_country_std_price', 'user_country_mean_spending',
                            'user_country_std_spending', 'hotel_position_mean',
                            'hotel_position_std', 'number_of_nans']

    if normalize:
        # normalize with unit gaussian centered
        for variable in things_to_normalize:
            df[variable]=(df[variable]-df[variable].mean())/df[variable].var()

    else:
        # scale between []-1, 1]
        for variable in things_to_normalize:
            df[variable]=(df[variable]-df[variable].mean())/(df[variable].max()-df[variable].min())

    return df

def count_nan_feature(df):
    df["number_of_nans"] = df.isnull().sum(axis=1)
    return df

def add_relevance_labels (df):
    def relevance (x):
        if x['booking_bool'] == 1:
            return 5
        if x['click_bool'] == 1:
            return 1
        return 0
    df['relevance'] = df.apply(relevance, axis=1)
    return df

def drop_non_features(df, set_type):
    df.reset_index(inplace=True)
    shit_to_drop = [
        'srch_id',
        'prop_id',
        'site_id',
        'visitor_location_country_id',
        'prop_country_id',
        'srch_destination_id',
        'country_cluster',
        'time_of_check_in',
        'time_of_check_out',
        'date_time'
    ]

    if set_type == "training" or set_type == "validation":
        shit_to_drop += ['booking_bool',
                         'click_bool',
                         'position',
                         'relevance',
                         'gross_bookings_usd']
    df.drop(shit_to_drop, axis=1, inplace=True)
    return df

def apply_PCA (df, set_type):
    matrix = df.to_numpy()
    if set_type == "training":
        # normalize data
        scaler = StandardScaler()
        scaler.fit(matrix)
        normalized_matrix = scaler.transform(matrix)

        # apply PCA
        pca = PCA(.95)
        pca.fit(normalized_matrix)
        projected_matrix = pca.transform(normalized_matrix)

        # save normalization and PCA transformations
        with open('PCA.pkl', 'wb') as f:
            pickle.dump((scaler, pca), f)
    else:
        with open('PCA.pkl', 'rb') as f:
            scaler, pca = pickle.load(f)
            normalized_matrix = scaler.transform(matrix)
            projected_matrix = pca.transform(normalized_matrix)

    print(f"PCA reduced {matrix.shape[1]} dimensions to {pca.n_components_} components "
          f"and retained {pca.explained_variance_ratio_.sum() * 100.0:.2f}% of the variance.")
    return projected_matrix

def prepare_for_mart(path, set_type):
    df = load_the_datas(path, set_type=set_type)
    df = preprocess(df, set_type=set_type)

    # srch_ids = df.index.get_level_values('srch_id').to_numpy()
    df.reset_index(inplace=True)
    srch_ids = df['srch_id'].to_numpy()
    prop_ids = df['prop_id'].to_numpy()
    relevancies = np.array(1)
    if set_type == "training" or set_type == "validation":
        relevancies = df['relevance'].to_numpy()

    df = drop_non_features(df, set_type=set_type)

    features = apply_PCA(df, set_type=set_type)
    return features, relevancies, srch_ids, prop_ids


def preprocess(df, set_type):
    df = average_competitors(df)
    df = count_nan_feature(df)
    df = remove_nans(df)
    df = outlier_killer(df)
    df = cluster_hotel_countries(df)
    df = add_circular_day_of_year_features(df)
    df = add_seasons(df)
    df = id_hacking(df)

    # df = normalizer(df) its broken and I don't want to fix it :)
    if set_type == "training" or set_type == "validation":
        df = add_relevance_labels(df)
        # df = balance_sampling(df)
        # df = balance_relevancies(df)

    return df

def make_plots (df):
    booking_days(df)
    polar_booking_days(df)
    showNaNs(df)
    box_plot_variable(df)
    show_me_the_money(df)
    do_you_like_pie(df)
    country_price_vagina_plots(df)
    # scatterplotter(df) # VERY HEAVY DO NOT RUN
    correlation_matrixo(df)
    show_country_clusters(df)
    polar_graph(df)
    histogram_site_id(df)
    # position_bias(df)

if __name__ == "__main__":
    path = '../data/tiny_train.csv'

    # x, y, srch_ids, prop_ids = prepare_for_mart(path)
    #
    # exit(-1)
    # path = '../data/tenth_train.csv'

    parser = argparse.ArgumentParser(prog='Datamining techniques assignment 1 (advanced)')
    parser.add_argument('--force_preprocess', action='store_true')
    args = parser.parse_args()

    preprocessed_dataset_file = Path("preprocessed_data.pkl")
    if not preprocessed_dataset_file.exists() or args.force_preprocess:
        df = load_the_datas(path, set_type="training")
        df = preprocess(df, set_type="training")

        file_stream = open(preprocessed_dataset_file, 'wb')
        pickle.dump(df, file_stream)
        print(f'Wrote preprocessed dataset to \'{preprocessed_dataset_file}\'.')
    else:
        file_stream = open(preprocessed_dataset_file, 'rb')
        df = pickle.load(file_stream)
        print(f'Loaded preprocessed dataset from \'{preprocessed_dataset_file}\'.')

    # make_plots(df)

    # df = normalizer(df)
    df.info()
    print("done")
