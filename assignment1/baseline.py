import argparse
import pickle
from pathlib import Path
from random import shuffle
from typing import Tuple, Any

from ESN import try_ESN
from visualizations import *


def load_data(filename='dataset_mood_smartphone.csv'):
    types = {'id': str, 'time': str, 'variable': str, 'value': float}
    parse_dates = ['time']
    df = pd.read_csv(filename, index_col=0,
                     dtype=types, parse_dates=parse_dates)

    df = pd.pivot_table(df, index=['id', 'time'], columns='variable', values='value')

    return df

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


def calculate_baseline_previous_day(df):

    counter = 0
    loss = 0
    for id, new_df in df.groupby(level='id'):
        previous_value = np.nan
        for time, mood in new_df.groupby(level='time'):
            if (not np.isnan(mood.mood.values[0]) and not np.isnan(previous_value)):
                loss += abs(previous_value - mood.mood.values[0])**2
                counter += 1
            previous_value = mood.mood.values[0]
    print("RMSE based on previous day: %3.9f over %d datapoints" % (
        (loss / counter) ** .5, counter))


def calculate_baseline_mean_global(df):
    counter = 0
    loss = 0

    df = df[df["mood"].notnull()]
    mean_mood = df["mood"].mean()

    for row in df["mood"]:
        loss += abs(row - mean_mood) ** 2
        counter += 1

    print("RMSE based on global average: %3.9f over %d datapoints" % (
        (loss / counter) ** .5, counter))


def calculate_baseline_mean_per_person(df):
    counter = 0
    loss = 0

    df = df[df["mood"].notnull()]
    for id, new_df in df.groupby(level="id"):
        mean_mood = new_df["mood"].mean()
        for row in new_df["mood"]:
            loss += abs(row - mean_mood) ** 2
            counter += 1
    print("RMSE based on person average: %3.9f over %d datapoints" % (
        (loss / counter) ** .5, counter))

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

    to_interpolate_linear = ['activity']

    df['mood_interpolated'] = df['mood']
    to_interpolate_pad = ['mood_interpolated', 'moodDeviance', 'circumplex.arousalDeviance',
                             'circumplex.valenceDeviance', 'circumplex.arousal',
                             'circumplex.valence']

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
        df[variable]=(df[variable]-df[variable].mean())/df[variable].var()
        # df[variable]=(df[variable]-df[variable].mean())/(df[variable].max()-df[variable].min())


    time_variables = ['screen', 'appCat.builtin', 'appCat.communication',
    'appCat.entertainment', 'appCat.finance', 'appCat.game',
    'appCat.office', 'appCat.other', 'appCat.social', 'appCat.travel',
    'appCat.unknown', 'appCat.utilities', 'appCat.weather']

    for variable in time_variables:
        df[variable]=(df[variable]-df[variable].mean())/df[variable].var()
        # df[variable]=(df[variable]-df[variable].mean())/(df[variable].max()-df[variable].min())
    return df

def remove_wrong_data(df):
    time_variables = ['screen', 'appCat.builtin', 'appCat.communication',
    'appCat.entertainment', 'appCat.finance', 'appCat.game',
    'appCat.office', 'appCat.other', 'appCat.social', 'appCat.travel',
    'appCat.unknown', 'appCat.utilities', 'appCat.weather', 'call', 'sms']

    for variable in time_variables:
        df[variable][df[variable] < 0] = 0
    return df

def remove_after_date(df, date='2014-05-5'):

    # print(df.loc[(slice(None), pd.date_range(start='2014-02-17', end=date)), :])

    return df.loc[(slice(None), pd.date_range(start='2014-02-17', end=date)), :]

def create_instance_dataset(dataset) -> Tuple[Any, Any, pd.DataFrame, float]:
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

def remove_before_first_target (df):
    # df_no_mood = df.loc[:, df.columns != 'mood']
    # thing = df_no_mood.dropna()
    # thing['mood'] = df
    return df.dropna(subset=df.columns.drop('mood'))

def split_dataset_by_person (dataset, test_fraction=0.2):
    ding = dataset.groupby(level='id')
    split = [(id, new_df) for id, new_df in ding]
    shuffle(split)
    splitpoint = int(len(split) * (1-test_fraction))
    training_set = split[:splitpoint]
    test_set = split[splitpoint:]
    return training_set, test_set

if __name__ == "__main__":
    base_seed = 47524
    import numpy as np
    import torch
    import random
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)
    random.seed(base_seed)

    parser = argparse.ArgumentParser(prog='Datamining techniques assignment 1 (advanced)')
    parser.add_argument('--force_preprocess', action='store_true')
    args = parser.parse_args()

    preprocessed_dataset_file = Path("preprocessed_data.pkl")
    if not preprocessed_dataset_file.exists() or args.force_preprocess:
        print('Loading and preprocessing data.')
        df = load_data('dataset_mood_smartphone.csv')
        df = aggregate_into_days(df)
        df = reindex_to_days(df)
        df = replace_nans_with_zeros(df)
        df = calculate_deviance(df)
        df = interpolate_data(df)
        df = add_time_features(df)
        # df = normalize_minutes(df)
        df = remove_wrong_data(df)
        df = normalize_dataset(df)
        df = remove_before_first_target(df)
        df = remove_after_date(df, date='2014-05-5')
        file_stream = open(preprocessed_dataset_file, 'wb')
        pickle.dump(df, file_stream)
        print(f'Wrote preprocessed dataset to \'{preprocessed_dataset_file}\'.')
    else:
        file_stream = open(preprocessed_dataset_file, 'rb')
        df = pickle.load(file_stream)
        print(f'Loaded preprocessed dataset from \'{preprocessed_dataset_file}\'.')

    calculate_baseline_previous_day(df)
    calculate_baseline_mean_global(df)
    calculate_baseline_mean_per_person(df)
    # daan_frame = create_instance_dataset(df)
    # print(daan_frame[0])
    # box_plot_id(df)
    # box_plot_variable(df)

    n_random_tests = 100
    training_RMSE_values = []
    test_RMSE_values = []
    n_hiddens = [1, 2, 5, 10, 20, 50, 100, 200]
    for random_i in range(n_random_tests):
        print(f"Randomly seeded run {random_i+1}/{n_random_tests}")
        np.random.seed(base_seed + random_i)
        torch.manual_seed(base_seed + random_i)
        random.seed(base_seed + random_i)
        mood_mean, mood_min, mood_max, mood_var = df['mood'].mean(), df['mood'].min(), df['mood'].max(), df['mood'].var()
        training_set, test_set = split_dataset_by_person(df)

        training_RMSE_values.append([])
        test_RMSE_values.append([])
        training_RMSE_values[random_i] = []
        test_RMSE_values[random_i] = []
        for n in n_hiddens:
            training_RMSE, test_RMSE = try_ESN(training_set, test_set, n, mood_mean, mood_min, mood_max, mood_var)
            training_RMSE_values[random_i].append(training_RMSE)
            test_RMSE_values[random_i].append(test_RMSE)
            # print(f"ESN ({n}) RMSE training set: {training_RMSE}")
            # print(f"ESN ({n}) RMSE test set: {test_RMSE}")

    np.savetxt(f"ESN_{n_random_tests}_runs_training_RMSE.csv", training_RMSE_values, delimiter=";")
    np.savetxt(f"ESN_{n_random_tests}_runs_test_RMSE.csv", test_RMSE_values, delimiter=";")

    avg_training_RMSE_per_size = np.array(training_RMSE_values).mean(axis=0)
    avg_test_RMSE_per_size = np.array(test_RMSE_values).mean(axis=0)
    best_training_RMSE = avg_training_RMSE_per_size.min()
    best_training_size = n_hiddens[avg_training_RMSE_per_size.argmin()]
    best_test_RMSE = avg_test_RMSE_per_size.min()
    best_test_size = n_hiddens[avg_test_RMSE_per_size.argmin()]
    print(f"Best model on training: size of {best_training_size} with RMSE of {best_training_RMSE}")
    print(f"Best model on test: size of {best_test_size} with RMSE of {best_test_RMSE}")

    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.title(f"Echo state network ({n_random_tests} run average)")
    plt.plot(n_hiddens, np.array(training_RMSE_values).mean(axis=0), marker='x', label='training error')
    plt.plot(n_hiddens, np.array(test_RMSE_values).mean(axis=0), marker='x', label='test error')
    plt.xscale('log')
    plt.xlabel('Reservoir size (log)')
    plt.yscale('log', nonposy='clip')
    plt.ylabel('RMSE (log)')
    ax = plt.gca()
    # ax.set_ylim(top=1.1)
    ax.set_yticks([0.7, 0.8, 1.0])
    plt.legend()
    plt.show(block=True)
    # box_plot(df)
    # thing(df)
    # scatterplot_mood(df)
    # print(training_set[0][1].info())
    # print(training_set[0][1].head())

    # correlation_matrix(df)
    # scatter_matrix_plot(df)
