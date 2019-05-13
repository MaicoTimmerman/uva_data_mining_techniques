
def stats (x):
    return min(x), sum(x)/len(x), max(x)

def count_hotels_per_search (dataset):
    df = dataset[['srch_id', 'price_usd']]
    hotels_per_search = [len(df2['price_usd']) for srch_id, df2 in df.groupby('srch_id')]
    return stats(hotels_per_search)

def count_clicks_per_search (dataset):
    df = dataset[['srch_id', 'click_bool']]
    clicks_per_search = [df2['click_bool'].sum() for srch_id, df2 in df.groupby('srch_id')]
    return stats(clicks_per_search)

def count_bookings_per_search (dataset):
    df = dataset[['srch_id', 'booking_bool']]
    bookings_per_search = [df2['booking_bool'].sum() for srch_id, df2 in df.groupby('srch_id')]
    return stats(bookings_per_search)

def count_clicks_per_booking (dataset):
    df = dataset[['booking_bool', 'click_bool']]
    clicks_per_booking = df[df['booking_bool'] == True]['click_bool']
    return stats(clicks_per_booking)

def calc_price_per_booking (dataset):
    df = dataset[['booking_bool', 'price_usd']]
    clicks_per_booking = df[df['booking_bool'] == True]['price_usd']
    return stats(clicks_per_booking)

def count_countries_per_search (dataset):
    df = dataset[['srch_id', 'prop_country_id']]
    countries_per_search = [len(df2['prop_country_id'].unique()) for srch_id, df2 in df.groupby('srch_id')]
    return stats(countries_per_search)

def count_hotel_countries (dataset):
    return len(dataset['prop_country_id'].unique())

def count_visitor_countries (dataset):
    return len(dataset['visitor_location_country_id'].unique())

def count_destinations (dataset):
    return len(dataset['srch_destination_id'].unique())

def count_countries_from_which_bookings_were_made (dataset):
    df = dataset[['booking_bool', 'visitor_location_country_id']]
    countries_from_which_bookings_were_made = df[df['booking_bool'] == True]['visitor_location_country_id'].unique()
    return len(countries_from_which_bookings_were_made)

def count_random_bools (dataset):
    df = dataset[['srch_id', 'random_bool']]
    random_bools = [df2['random_bool'].iloc[0] for srch_id, df2 in df.groupby('srch_id')]
    return sum(random_bools)