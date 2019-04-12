import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import argparse
from random import shuffle
from visualizations import *
def load_data(filename='dataset_mood_smartphone.csv'):
    types = {'id': str, 'time': str, 'variable': str, 'value': float}
    parse_dates = ['time']
    df = pd.read_csv(filename, index_col=0,
                     dtype=types, parse_dates=parse_dates)

    df = pd.pivot_table(df, index=['id', 'time'], columns='variable', values='value')

    def aggregate_into_days(df):
        to_mean_list = ["mood", "circumplex.arousal", "circumplex.valence", "activity"]

        level_values = df.index.get_level_values
        d = df.groupby([level_values(i) for i in [0]]
                              +[pd.Grouper(freq='D', level=-1)])

        summed = (d.agg('sum', axis=1))
        meaned = (d.apply(lambda x: x.mean()))

        # for to_mean in to_mean_list:
        #     summed[to_mean] = meaned[to_mean]
        summed[to_mean_list] = meaned[to_mean_list]
        return summed

    def reindex_to_days(df):

        date_range = pd.date_range(
            start="2014-02-17",
            end="2014-06-9",
            freq='D'
            )

        unique_id = df.index.unique(level='id')

        blank_dataframe = (
            pd.MultiIndex
            .from_product(
                iterables=[unique_id, date_range],
                names=['id', 'time']
            )
        )
        return df.reindex(blank_dataframe)


    df = aggregate_into_days(df)

    df = reindex_to_days(df)

    return df

def calculate_baseline(df):

    counter = 0
    loss = 0
    for id, new_df in df.groupby(level='id'):
        previous_value = np.nan
        for time, mood in new_df.groupby(level='time'):
            if (not np.isnan(mood.mood.values[0]) and not np.isnan(previous_value)):
                loss += abs(previous_value - mood.mood.values[0])**2
                counter += 1
            previous_value = mood.mood.values[0]
    print("Average loss (RMSE) %3.9f over %d datapoints" % ((loss / counter)**.5, counter))

def replace_nans_with_zeros (df):
    to_fix = ['appCat.builtin', 'appCat.communication',
       'appCat.entertainment', 'appCat.finance', 'appCat.game',
       'appCat.office', 'appCat.other', 'appCat.social', 'appCat.travel',
       'appCat.unknown', 'appCat.utilities', 'appCat.weather', 'call', 'screen', 'sms']
    df[to_fix] = df[to_fix].fillna(0)
    return df

def interpolate_data(df):

    # to_interpolate_linear = ['activity', 'appCat.builtin', 'appCat.communication',
    #    'appCat.entertainment', 'appCat.finance', 'appCat.game',
    #    'appCat.office', 'appCat.other', 'appCat.social', 'appCat.travel',
    #    'appCat.unknown', 'appCat.utilities', 'appCat.weather', 'call',
    #    'circumplex.arousal', 'circumplex.valence', 'screen', 'sms',]

    to_interpolate_linear = ['moodDeviance', 'circumplex.arousalDeviance',
                             'circumplex.valenceDeviance', 'circumplex.arousal',
                             'circumplex.valence', 'activity']

    df['mood_interpolated'] = df['mood']
    to_interpolate_pad = ['mood_interpolated']

    for id in df.index.unique(level='id'):
        for variable in to_interpolate_linear:
             # this is so ugly but pandas has this shit issue at its core
             #  SEE: https://www.dataquest.io/blog/settingwithcopywarning/
            df.loc[(id, slice(None))][variable] =\
            df.loc[(id, slice(None))][variable].interpolate(method='linear', limit_direction='forward')

        for variable in to_interpolate_pad:
            df.loc[(id, slice(None))][variable] =\
            df.loc[(id, slice(None))][variable].interpolate(method='pad', limit_direction='forward', limit=None)
    # print(df.loc[('AS14.01', slice(None)), 'mood'])
    return df

def calculate_deviance(df):

    # to_calculate_deviance = ['activity', 'appCat.builtin', 'appCat.communication',
    #    'appCat.entertainment', 'appCat.finance', 'appCat.game',
    #    'appCat.office', 'appCat.other', 'appCat.social', 'appCat.travel',
    #    'appCat.unknown', 'appCat.utilities', 'appCat.weather', 'call',
    #    'circumplex.arousal', 'circumplex.valence', 'mood', 'screen', 'sms',]

    to_calculate_deviance = ['mood', 'circumplex.arousal', 'circumplex.valence']

    for variable in to_calculate_deviance:
        variable_Deviance = variable + 'Deviance'
        df[variable_Deviance] = np.nan

    for id in df.index.unique(level='id'):
        for variable in to_calculate_deviance:
            variable_Deviance = variable + 'Deviance'
            variable_average = df.loc[(id, slice(None))][variable].mean()

            df.loc[(id, slice(None))][variable_Deviance] =\
            df.loc[(id, slice(None))][variable].apply(lambda x: x - variable_average)


    # print(df.loc[('AS14.01', slice(None)), 'moodDeviance'])
    # print(df.loc[('AS14.01', slice(None)), 'mood'])
    return df

def add_time_features(df):

    df['WEEKEND'] = ((pd.DatetimeIndex(df.index.get_level_values("time")).dayofweek) // 5 == 1).astype(int)
    df['MONDAY'] = ((pd.DatetimeIndex(df.index.get_level_values("time")).dayofweek) == 0).astype(int)

    df['SPRING'] = np.where((pd.DatetimeIndex(df.index.get_level_values("time")).month).isin([3,4,5]), 1, 0)
    df['SUMMER'] = np.where((pd.DatetimeIndex(df.index.get_level_values("time")).month).isin([6,7,8]), 1, 0)
    # There are no autumn days in our dataset.
    # df['AUTUMN'] = np.where((pd.DatetimeIndex(df.index.get_level_values("time")).month).isin([9,10,11]), 1, 0)
    df['WINTER'] = np.where((pd.DatetimeIndex(df.index.get_level_values("time")).month).isin([12,1,2]), 1, 0)
    return df

def normalize_minutes(df):
    time_variables = ['screen', 'appCat.builtin', 'appCat.communication',
    'appCat.entertainment', 'appCat.finance', 'appCat.game',
    'appCat.office', 'appCat.other', 'appCat.social', 'appCat.travel',
    'appCat.unknown', 'appCat.utilities', 'appCat.weather']

    seconds_in_day = 86400

    for variable in time_variables:
        # variable_new_name = variable + '_normalized'
        variable_new_name = variable
        df[variable_new_name] = df[variable].apply(lambda x: x/seconds_in_day)

    return df

def normalize_dataset(df):

    non_time_variables = ['call', 'sms', 'activity',
    'circumplex.arousal', 'circumplex.valence', 'sms', 'moodDeviance',
    'circumplex.arousalDeviance', 'circumplex.valenceDeviance', 'mood_interpolated']

    for variable in non_time_variables:
        # df[variable]=(df[variable]-df[variable].mean())/df[variable].std()
        df[variable]=(df[variable]-df[variable].mean())/(df[variable].max()-df[variable].min())


    time_variables = ['screen', 'appCat.builtin', 'appCat.communication',
    'appCat.entertainment', 'appCat.finance', 'appCat.game',
    'appCat.office', 'appCat.other', 'appCat.social', 'appCat.travel',
    'appCat.unknown', 'appCat.utilities', 'appCat.weather']

    for variable in time_variables:
        df[variable]=(df[variable]-df[variable].mean())
    return df

def remove_wrong_data(df):
    time_variables = ['screen', 'appCat.builtin', 'appCat.communication',
    'appCat.entertainment', 'appCat.finance', 'appCat.game',
    'appCat.office', 'appCat.other', 'appCat.social', 'appCat.travel',
    'appCat.unknown', 'appCat.utilities', 'appCat.weather', 'call', 'sms']

    for variable in time_variables:
        df[variable][df[variable] < 0] = 0
    return df

def create_instance_dataset(dataset):
    n_days = 3
    instance_dataset = []
    for person in dataset.index.unique(level='id'):
        # series = dataset.loc[(person, slice(None))].unstack()
        series = dataset.loc[(person, slice(None))]
        days_count = len(series)
        series = series.transpose()
        for i in range(days_count-n_days):
            target_day = series.columns[i+n_days]
            target = series.loc['mood', target_day]
            interval = series.loc[:, series.columns[i:i+n_days]]
            instance_dataset.append((person, target_day, interval.drop(labels='mood'), target))
    return instance_dataset

def split_dataset_by_person (dataset, test_fraction=0.2):
    ding = df.groupby(level='id')
    split = [(id, new_df) for id, new_df in ding]
    shuffle(split)
    splitpoint = int(len(split) * (1-test_fraction))
    training_set = split[:splitpoint]
    test_set = split[splitpoint:]
    return training_set, test_set

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Datamining techniques assignment 1 (advanced)')
    parser.add_argument('--force_preprocess', action='store_true')
    args = parser.parse_args()

    preprocessed_dataset_file = Path("preprocessed_data.pkl")
    if not preprocessed_dataset_file.exists() or args.force_preprocess:
        print('Loading and preprocessing data.')
        df = load_data()
        df = replace_nans_with_zeros(df)
        df = calculate_deviance(df)
        df = interpolate_data(df)
        df = add_time_features(df)
        df = normalize_minutes(df)
        df = remove_wrong_data(df)
        df = normalize_dataset(df)
        file_stream = open(preprocessed_dataset_file, 'wb')
        pickle.dump(df, file_stream)
        print(f'Wrote preprocessed dataset to \'{preprocessed_dataset_file}\'.')
    else:
        file_stream = open(preprocessed_dataset_file, 'rb')
        df = pickle.load(file_stream)
        print(f'Loaded preprocessed dataset from \'{preprocessed_dataset_file}\'.')

    calculate_baseline(df)
    daan_frame = create_instance_dataset(df)

    training_set, test_set = split_dataset_by_person(df)
    print(daan_frame[0])
    # box_plot_id(df)
    box_plot_variable(df)
    # thing(df)
    # print(training_set[0][1].info())
    # print(training_set[0][1].head())

    # correlation_matrix(df)
    # scatter_matrix_plot(df)
