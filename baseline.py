import pandas as pd
import numpy as np

def load_data(filename='dataset_mood_smartphone.csv'):
    types = {'id': str, 'time': str, 'variable': str, 'value': float}
    parse_dates = ['time']
    df = pd.read_csv(filename, index_col=0,
                     dtype=types, parse_dates=parse_dates)

    df.set_index(['id', 'variable', 'time'], inplace=True)

    def using_Grouper(df):
        level_values = df.index.get_level_values
        a = [level_values(i) for i in [0,1]]
        b = [pd.Grouper(freq='D', level='time')]
        c = [pd.Grouper(level='variable')]
        c = df.groupby(a+b)
        d = c.agg([np.sum, np.mean])
        return d

    # Now we create a multi-index. We set the hierarchy to be: ID -> Time ->
    # variables: value]

    df_by_day = using_Grouper(df)

    to_mean = ["mood", "circumplex.arousal", "circumplex.valence"]

    df_by_day['value2'] = np.where(df_by_day.index.isin(to_mean, level='variable'), df_by_day['value']['mean'], df_by_day['value']['sum'])
    # df_by_day.loc[("AS14.01", "mood")]
    df_by_day.drop('value', axis=1, inplace=True)
    df_by_day.rename(columns={'value2':'value'}, inplace=True)
    print(df_by_day.info())

    temper = df_by_day.reset_index()
    temper.set_index(['id', 'time', 'variable'], inplace=True)
    temper.sort_index(level=["id", "time"], inplace=True)

    date_range = pd.date_range(
    start="2014-02-17",
    end="2014-06-9",
    freq='D'
    )

    def blanker(temper):
        date_range = temper.index.unique(level='time')
        unique_id = temper.index.unique(level='id')
        unique_variables = temper.index.unique(level='variable')

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
    temper.columns = temper.columns.get_level_values(2)
    return temper

    # df_by_day = using_Grouper(df)
    # df_by_day = df_by_day.reset_index()
    # df_by_day.set_index(['id', 'time', 'variable'], inplace=True)
    # return df_by_day.sort_index(level=["id", "time"])


def calculate_baseline(df: pd.DataFrame):
    df_mood = df.loc[(slice(None), slice(None), ["mood"]), :]

    loss = 0
    counter = 0
    for index, row in df_mood.iterrows():
        index_prev_day = (index[0], index[1] + pd.DateOffset(-1), "mood")
        if index_prev_day in df_mood.index:
            loss += abs(row.value - df_mood.loc[index_prev_day].value)
            counter += 1

    print("Average loss %f over %d datapoints" % (loss / counter, counter))


def fill_df_with_nans(df):
    """
    handling missing missing values

    To create data with added missing values (if we wanted for example to have
    evenly sized vectors for each day) we would have to make an empty
    multi-index as above with all possible combinations and set all of their
    values to NaN.

    I follow these instructions to do this:
    https://medium.com/when-i-work-data/using-pandas-multiindex-from-product-to-fill-in-missing-data-43c3cfe9cf39
    """
    date_range = pd.date_range(start="2014-02-17",
                               end="2014-06-9",
                               freq='D')

    unique_id = df.index.unique(level='id')
    unique_variables = df.index.unique(level='variable')

    blank_dataframe = (
        pd.MultiIndex
        .from_product(
            iterables=[unique_id, date_range, unique_variables],
            names=['id', 'time', 'variable']
        )
    )
    blank_dataframe.values[:5]

    return df.reindex(blank_dataframe)

def create_instance_dataset(dataset):
    n_days = 3
    instance_dataset = []
    for person in dataset.index.unique(level='id'):
        series = dataset.loc[(person, slice(None))].unstack()
        days_count = len(series)
        series = series.transpose()
        for i in range(days_count-n_days):
            target_day = series.columns[i+n_days]
            target = series.loc[('value', '', 'mood'), target_day]
            interval = series.loc[:, series.columns[i:i+n_days]]
            instance_dataset.append((person, target_day, interval, target))
    return instance_dataset


if __name__ == "__main__":
    df = load_data()

    # calculate_baseline(df)
    # daan_frame = create_instance_dataset(df)
    print(df.info())
    print(df.index)
