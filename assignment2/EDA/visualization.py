import pandas as pd
from datetime import datetime
from datetime import timedelta
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt


def scatterplotter(df):
    to_do = [
    'visitor_hist_starrating',
    'visitor_hist_adr_usd',
    'prop_starrating',
    'prop_review_score',
    'prop_location_score1',
    'prop_location_score2',
    'prop_log_historical_price',
    'price_usd',
    'srch_destination_id',
    'srch_length_of_stay',
    'srch_booking_window',
    'srch_adults_count',
    'srch_children_count',
    'srch_room_count',
    'srch_query_affinity_score',
    'orig_destination_distance',]

    # axes = scatter_matrix(df[to_do], alpha=0.2, figsize=(6, 6), diagonal='kde')
    axes = scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='kde')

    for list in axes:
        for axe in list:
            plt.setp(axe.get_xticklabels(), visible=False)
            plt.setp(axe.get_yticklabels(), visible=False)
            axe.set_xlabel(axe.get_xlabel().split(".")[-1], rotation=45, rotation_mode='anchor', ha="right")
            axe.set_ylabel(axe.get_ylabel().split(".")[-1], rotation=0, rotation_mode='anchor', ha="right")

    plt.show()

def showNaNs(df):
    df.isna().sum().sort_values().plot(kind='bar', figsize=(20,20), logy=True)
    plt.show()

def correlation_matrixo(df):
    plt.matshow(df.corr())
    plt.show()

def show_country_clusters(df):

    scatter_data = df.groupby("prop_country_id").price_usd.describe()[['mean','std']]
    labels = df.groupby("prop_country_id").country_cluster.min()
    scatter_data = pd.concat([scatter_data, labels], axis=1)

    ax = scatter_data.plot(kind='scatter', x='mean', y='std', c='country_cluster', colormap='viridis')
    plt.show()
