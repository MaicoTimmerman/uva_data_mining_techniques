import pandas as pd
import pickle


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

def outlier_killer(df):
    q = df["price_usd"].quantile(0.999)
    # q = 4000
    # return df[(df["price_usd"] < q) & (df["booking_bool"] == 1)]
    return df[(df["price_usd"] < q)]

def make_generic_cheat_sheet(df, group_by_id, variables_to_average = None):

    df_to_use = df.loc[df['booking_bool'] == 1].groupby(group_by_id)

    if (variables_to_average == None):
        variables_to_average = ['srch_children_count', 'srch_adults_count',
                                'srch_booking_window', 'srch_length_of_stay',
                                'srch_query_affinity_score', 'orig_destination_distance',
                                'prop_location_score2', 'prop_location_score1',
                                'prop_review_score', 'prop_starrating', 'price_usd']

    name_for_std = group_by_id + '_' + 'price_usd' + '_std'
    cheat_cheet = df_to_use.price_usd.describe()[['std']]
    cheat_cheet.rename(columns = {"std": name_for_std}, inplace=True)

    for variable in variables_to_average:
        cheat_sheet_name = group_by_id + '_' + variable + '_mean'
        cheat_cheet[cheat_sheet_name] = df_to_use[variable].mean()

    cheat_cheet.fillna(0, inplace=True)
    return cheat_cheet

def prop_id_cheats_sheet(df):
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

    group_by_id = 'prop_id'

    name_for_std = group_by_id + '_' + 'position' + '_std'
    cheat_cheet = temp_df.groupby('prop_id')['position'].describe()[['std']]
    cheat_cheet.rename(columns = {"std": name_for_std}, inplace=True)

    cheat_cheet['prop_id_position_mean'] = temp_df.groupby('prop_id')['position'].describe()[['mean']]
    cheat_cheet['prop_id_book_ratio'] = temp_df.groupby('prop_id')['booking_bool'].sum()/temp_df.groupby('prop_id')['booking_bool'].count()
    cheat_cheet['prop_id_clicked_ratio'] = temp_df.groupby('prop_id')['click_bool'].sum()/temp_df.groupby('prop_id')['click_bool'].count()

    cheat_cheet.fillna(0, inplace=True)

    return cheat_cheet

if __name__ == "__main__":
    path = 'training_set_VU_DM.csv'
    # path = 'tiny_train.csv'
    df = load_the_datas(path)
    print("Data Loaded")

    df = outlier_killer(df)
    print("Outliers removed")

    cheat_sheet = prop_id_cheats_sheet(df)
    file_stream = open('cheat_sheet_prop_id.pkl', 'wb')
    pickle.dump(cheat_sheet, file_stream)
    print("prop_id_cheats_sheet made")

    group_by_id = 'visitor_location_country_id'
    cheat_sheet = make_generic_cheat_sheet(df, group_by_id)
    file_stream = open('cheat_sheet_visitor_location_country_id.pkl', 'wb')
    pickle.dump(cheat_sheet, file_stream)
    print("visitor_location_country_id_cheats_sheet made")

    group_by_id = 'site_id'
    cheat_sheet = make_generic_cheat_sheet(df, group_by_id)
    file_stream = open('cheat_sheet_site_id.pkl', 'wb')
    pickle.dump(cheat_sheet, file_stream)
    print("site_id_cheats_sheet made")

    group_by_id = 'srch_destination_id'
    cheat_sheet = make_generic_cheat_sheet(df, group_by_id)
    file_stream = open('cheat_sheet_srch_destination_id.pkl', 'wb')
    pickle.dump(cheat_sheet, file_stream)
    print("srch_destination_id_cheats_sheet made")

    group_by_id = 'prop_country_id'
    cheat_sheet = make_generic_cheat_sheet(df, group_by_id)
    file_stream = open('cheat_sheet_prop_country_id.pkl', 'wb')
    pickle.dump(cheat_sheet, file_stream)
    print("prop_country_id_cheats_sheet made")

    print("done")
