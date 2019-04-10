import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(filename='dataset_mood_smartphone.csv'):
    types = {'id': str, 'time': str, 'variable': str, 'value': float}
    parse_dates = ['time']
    df = pd.read_csv(filename, index_col=0,
                     dtype=types, parse_dates=parse_dates)


    df.set_index(['id', 'variable', 'time'], inplace=True)

    def using_Grouper(df):
        to_mean = ["mood", "circumplex.arousal", "circumplex.valence", "activity"]
        level_values = df.index.get_level_values
        a = [level_values(i) for i in [0,1]]
        b = [pd.Grouper(freq='D', level='time')]
        c = [pd.Grouper(level='variable')]
        c = df.groupby(a+b)
        d = c.apply(lambda x: np.where(x.index.isin(to_mean, level='variable'), x.mean(), x.sum())[0])
        return d

    df_by_day = using_Grouper(df)

    # we create a multi-index. We set the hierarchy to be: ID -> Time ->
    # variables: value]

    temper = df_by_day.reset_index()
    temper.set_index(['id', 'time', 'variable'], inplace=True)
    temper.sort_index(level=["id", "time"], inplace=True)

    def blanker(df):
        # date_range = df.index.unique(level='time')
        date_range = pd.date_range(
        start="2014-02-17",
        end="2014-06-9",
        freq='D'
        )
        unique_id = df.index.unique(level='id')
        unique_variables = df.index.unique(level='variable')

        blank_dataframe = (
            pd.MultiIndex
            .from_product(
                iterables=[unique_id, date_range, unique_variables],
                names=['id', 'time', 'variable']
            )
        )
        return blank_dataframe

    blank_dataframe = blanker(temper)
    temper = temper.reindex(blank_dataframe)
    temper = temper.unstack()
    temper.columns = temper.columns.get_level_values(1)

    # print(temper.info())
    return temper

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

    print("Average loss %3.9f over %d datapoints" % ((loss / counter)**.5, counter))

def interpolate_data(df):

    to_interpolate_linear = ['activity', 'appCat.builtin', 'appCat.communication',
       'appCat.entertainment', 'appCat.finance', 'appCat.game',
       'appCat.office', 'appCat.other', 'appCat.social', 'appCat.travel',
       'appCat.unknown', 'appCat.utilities', 'appCat.weather', 'call',
       'circumplex.arousal', 'circumplex.valence', 'screen', 'sms',]

    to_interpolate_linear += ['moodDeviance', 'circumplex.arousalDeviance', 'circumplex.valenceDeviance']

    to_interpolate_pad = ['mood']

    for id in df.index.unique(level='id'):
        for variable in to_interpolate_linear:
             # this is so ugly but pandas has this shit issue at its core
             #  SEE: https://www.dataquest.io/blog/settingwithcopywarning/
            df.loc[(id, slice(None))][variable] =\
            df.loc[(id, slice(None))][variable].interpolate(method='linear', limit_direction='forward')

        for variable in to_interpolate_pad:
            df.loc[(id, slice(None))][variable] =\
            df.loc[(id, slice(None))][variable].interpolate(method='pad', limit_direction='forward', limit = 5)
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
            instance_dataset.append((person, target_day, interval, target))
    return instance_dataset


def correlation_matrix(df):
    # https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html#sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py
    labels = list(map(lambda x: x.split(".")[-1], df.columns.values))
    fig, ax = plt.subplots()

    matrix = df.corr(method='pearson').values

    im = ax.imshow(matrix)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

    # for i in range(len(labels)):
    #     for j in range(len(labels)):
    #         text = ax.text(j, i, matrix[i, j],
    #                        ha="center", va="center", color="w")

    fig.tight_layout()
    plt.show()

def box_plot(df):
    ax = df.boxplot()

    labels = list(map(lambda x: x.split(".")[-1], df.columns.values))
    labels.insert(0, "")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
    plt.show()

def scatter_matrix_plot(df):
    # https://machinelearningmastery.com/visualize-machine-learning-data-python-pandas/
    from pandas.plotting import scatter_matrix

    # fig, ax = plt.subplots()
    #
    scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='kde')
    # print(type(scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='kde')))
    print(plt)
    # labels = list(map(lambda x: x.split(".")[-1], df.columns.values))
    # # ax.set_xticks(np.arange(len(labels)))
    # # ax.set_yticks(np.arange(len(labels)))
    # ax.set_xticklabels(labels)
    # ax.set_yticklabels(labels)
    #
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #      rotation_mode="anchor")

    # fig.tight_layout()
    # labels = list(map(lambda x: x.split(".")[-1], df.columns.values))
    # ax.set_xticks(np.arange(len(labels)))
    # ax.set_xticklabels(labels)
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #      rotation_mode="anchor")
    plt.show()

if __name__ == "__main__":
    df = load_data()

    df = calculate_deviance(df)
    df = interpolate_data(df)
    df = add_time_features(df)
    df = normalize_minutes(df)

    calculate_baseline(df)
    # daan_frame = create_instance_dataset(df)
    # print(daan_frame)
    # # print(df.describe())
    # correlation_matrix(df)
    # box_plot(df)
    #
    # df.hist()
    # # plt.axis('image')
    # # plt.tight_layout()
    # # plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    # # plt.tight_layout(pad = 1)
    # # plt.rcParams.update({'font.size': 2})
    # plt.show()
    # scatter_matrix_plot(df)

    # print(df.info())
    # print(df.columns)
