import pandas as pd


def load_data(filename='dataset_mood_smartphone.csv'):
    types = {'id': str, 'time': str, 'variable': str, 'value': float}
    parse_dates = ['time']
    df = pd.read_csv(filename, index_col=0,
                     dtype=types, parse_dates=parse_dates)

    df.set_index(['id', 'variable', 'time'], inplace=True)

    def using_Grouper(df):
        level_values = df.index.get_level_values
        return (df.groupby([level_values(i) for i in [0, 1]] +
                           [pd.Grouper(freq='D', level=-1)]).mean())

    # Now we create a multi-index. We set the hierarchy to be: ID -> Time ->
    # variables: value
    df_by_day = using_Grouper(df)
    df_by_day = df_by_day.reset_index()
    df_by_day.set_index(['id', 'time', 'variable'], inplace=True)
    return df_by_day.sort_index(level=["id", "time"])


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


if __name__ == "__main__":
    df = load_data()
    calculate_baseline(df)

