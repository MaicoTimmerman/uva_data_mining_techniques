import pandas as pd
from data_questions import *


if __name__ == "__main__":
    input_file = "../data/tiny_train.csv"
    dataset = pd.read_csv(input_file)
    print('(min, mean, max)')
    print(f"A search results in {clicks_per_search(dataset)} clicks.")
    print(f"A search results in {bookings_per_search(dataset)} bookings.")
    print(f"A booking has {clicks_per_booking(dataset)} clicks.")
