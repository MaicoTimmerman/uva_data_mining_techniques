import pandas as pd
from data_questions import *


if __name__ == "__main__":
    input_file = "../data/tiny_train.csv"
    dataset = pd.read_csv(input_file)

    clicks_per_search = count_clicks_per_search(dataset)
    bookings_per_search = count_bookings_per_search(dataset)
    clicks_per_booking = count_clicks_per_booking(dataset)
    price_per_booking = calc_price_per_booking(dataset)

    print('(min, mean, max)')
    print(f"A search results in {clicks_per_search} clicks.")
    print(f"A search results in {bookings_per_search} bookings.")
    print(f"A booking has {clicks_per_booking} clicks.")
    print(f"A booking costs {price_per_booking} usd.")

    does_ignored_search_exist = clicks_per_search[0] < 1
    does_plural_booking_exist = bookings_per_search[2] > 1
    does_ghost_booking_exist = clicks_per_booking[0] < 1
    does_negative_price_exist = price_per_booking[0] < 0
    is_there_a_problem = (
        does_ignored_search_exist
        | does_ghost_booking_exist
        | does_plural_booking_exist
    )
    if is_there_a_problem:
        for _ in range(10):
            print("UH OH")
    else:
        print("Seems okay")