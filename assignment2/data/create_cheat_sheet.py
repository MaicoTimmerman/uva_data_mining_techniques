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

def visitor_location_country_id_cheat_sheet(df):
    cheat_cheet = df.loc[df['booking_bool'] == 1].groupby("visitor_location_country_id").price_usd.describe()[['mean','std']]
    # mat = df.loc[df['visitor_hist_adr_usd'] > 0].groupby("visitor_location_country_id").visitor_hist_adr_usd.describe()[['mean','std']]
    cheat_cheet = cheat_cheet.fillna(0)

    return cheat_cheet

def site_id_cheat_sheet(df):
    cheat_cheet = df.loc[df['booking_bool'] == 1].groupby("site_id").price_usd.describe()[['mean','std']]
    cheat_cheet = cheat_cheet.fillna(0)

    return cheat_cheet

def srch_destination_id_cheat_sheet(df):
    cheat_cheet = df.loc[df['booking_bool'] == 1].groupby("srch_destination_id").price_usd.describe()[['mean','std']]
    cheat_cheet = cheat_cheet.fillna(0)

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

    booker_bools = temp_df.groupby('prop_id')['booking_bool'].sum()/temp_df.groupby('prop_id')['booking_bool'].count()
    clicker_bools = temp_df.groupby('prop_id')['click_bool'].sum()/temp_df.groupby('prop_id')['click_bool'].count()
    cheat_cheet = temp_df.groupby('prop_id')['position'].describe()[['mean','std']]

    cheat_cheet.fillna({'mean':0, 'std': 0}, inplace=True)

    cheat_cheet['click_ratio'] = clicker_bools
    cheat_cheet['book_ratio'] = booker_bools

    return cheat_cheet

if __name__ == "__main__":
    path = 'training_set_VU_DM.csv'
    df = load_the_datas(path)
    df = outlier_killer(df)

    cheat_sheet = prop_id_cheats_sheet(df)
    file_stream = open('cheat_sheet_prop_id.pkl', 'wb')
    pickle.dump(cheat_sheet, file_stream)

    cheat_sheet = visitor_location_country_id_cheat_sheet(df)
    file_stream = open('cheat_sheet_visitor_location_country_id.pkl', 'wb')
    pickle.dump(cheat_sheet, file_stream)

    cheat_sheet = site_id_cheat_sheet(df)
    file_stream = open('cheat_sheet_site_id.pkl', 'wb')
    pickle.dump(cheat_sheet, file_stream)

    cheat_sheet = srch_destination_id_cheat_sheet(df)
    file_stream = open('cheat_sheet_srch_destination_id.pkl', 'wb')
    pickle.dump(cheat_sheet, file_stream)

    print("done")
