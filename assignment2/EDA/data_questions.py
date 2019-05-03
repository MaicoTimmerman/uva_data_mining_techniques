
def stats (x):
    return min(x), sum(x)/len(x), max(x)

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