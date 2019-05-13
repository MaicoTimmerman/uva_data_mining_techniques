import pandas as pd
from datetime import datetime
from datetime import timedelta
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from math import radians

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
    ax = df.isna().sum().sort_values().plot(kind='bar', figsize=(20,20), logy=True)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

    # plt.tight_layout()
    # plt.autoscale()
    plt.gcf().subplots_adjust(bottom=0.2)
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

def box_plot_variable(df):

    to_drop = ['prop_id', 'site_id', 'visitor_location_country_id', 'prop_country_id', 'orig_destination_distance',
                'srch_destination_id']
                # ,'srch_query_affinity_score', 'srch_saturday_night_bool']

    df_boxplot = df.drop(columns=to_drop)
    ax = df_boxplot.boxplot(showfliers=False, )

    # labels = list(map(lambda x: x.split(".")[-1], df.columns.values))
    # labels.insert(0, "")

    # labels = df.columns
    # ax.set_xticks(np.arange(len(labels)))
    # ax.set_xticklabels(labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
    # fig.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.show()

def show_me_the_money(df):

    df.groupby("booking_bool").price_usd.plot.hist(logy=True, sharex=True, sharey=True, range=(0,3000), bins=20)
    # pd.DataFrame.hist(  data=df, column='price_usd', by='booking_bool',
    #                     log=True, stacked=True, sharex=True, sharey=True)
    L=plt.legend()
    L.get_texts()[0].set_text('No booking')
    L.get_texts()[1].set_text('Booking')
    plt.title("Price of hotels seperated by booking")
    plt.xlabel("price_usd")
    plt.ylabel("Count")
    plt.show()

def do_you_like_pie(df):

    # print(df.columns)
    df_pie = df.reset_index()

    df.loc[df['booking_bool'] == 1].groupby("visitor_location_country_id").price_usd.describe()[['mean']].plot.pie(subplots=True)
    plt.title("Mean price of booked hotels per country")
    plt.show()
    df.loc[df['visitor_hist_adr_usd'] > 0].groupby("visitor_location_country_id").visitor_hist_adr_usd.describe()[['mean']].plot.pie(subplots=True)
    plt.title("Mean price of historicly booked hotels per country")
    plt.show()

def polar_graph(df):

    df = df[df['booking_bool'] == 1]

    if False:
        data_mean = df.groupby(by=[pd.DatetimeIndex(df.time_of_check_in).month]).mean()
        data_sum = df.groupby(by=[pd.DatetimeIndex(df.time_of_check_in).month]).sum()
        data_count = df.groupby(by=[pd.DatetimeIndex(df.time_of_check_in).month]).count()
        data_sum.loc[13] = data_sum.loc[1]
        data_mean.loc[13] = data_mean.loc[1]
        data_count.loc[13] = data_count.loc[1]

        data_mean = data_mean.reindex(list((v) for v in range(data_mean.index[0], data_mean.index[-1]+1))).fillna(0)
        data_sum = data_sum.reindex(list((v) for v in range(data_sum.index[0], data_sum.index[-1]+1))).fillna(0)
        data_count = data_count.reindex(list((v) for v in range(data_count.index[0], data_count.index[-1]+1))).fillna(0)
    if True:
        data_mean = df.groupby(by=[pd.DatetimeIndex(df.time_of_check_in).week]).mean()
        data_sum = df.groupby(by=[pd.DatetimeIndex(df.time_of_check_in).week]).sum()
        data_count = df.groupby(by=[pd.DatetimeIndex(df.time_of_check_in).week]).count()
        data_std = df.groupby(by=[pd.DatetimeIndex(df.time_of_check_in).week]).std()

        data_sum.loc[53] = data_sum.loc[1]
        data_mean.loc[53] = data_mean.loc[1]
        data_count.loc[53] = data_count.loc[1]
        data_std.loc[53] = data_std.loc[1]

        data_mean = data_mean.reindex(list((v) for v in range(data_mean.index[0], data_mean.index[-1]+1))).fillna(0)
        data_sum = data_sum.reindex(list((v) for v in range(data_sum.index[0], data_sum.index[-1]+1))).fillna(0)
        data_count = data_count.reindex(list((v) for v in range(data_count.index[0], data_count.index[-1]+1))).fillna(0)
        data_std = data_std.reindex(list((v) for v in range(data_std.index[0], data_std.index[-1]+1))).fillna(0)

    theta = np.linspace(0,2*np.pi,len(data_mean.index))
    ticks = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']

    ax = plt.subplot(221, projection='polar')

    ax.plot(theta, data_count['price_usd'].values, 'r')
    ax.set_xticklabels(ticks)
    ax.set_xticks(np.linspace(0,2*np.pi,13))
    ax.fill_between(theta, data_count['price_usd'].values, alpha=0.2, color='red')
    plt.title("Total amount of bookings")

    ax = plt.subplot(222, projection='polar')

    ax.plot(theta, data_mean['price_usd'].values, 'r')
    ax.set_xticklabels(ticks)
    ax.set_xticks(np.linspace(0,2*np.pi,13))
    ax.fill_between(theta, data_mean['price_usd'].values, alpha=0.2, color='red')

    plt.title("Average amount of monies spent on bookings")

    ax = plt.subplot(223, projection='polar')

    ax.plot(theta, data_sum['price_usd'].values, 'r')
    ax.set_xticklabels(ticks)
    ax.set_xticks(np.linspace(0,2*np.pi,13))
    ax.fill_between(theta, data_sum['price_usd'].values, alpha=0.2, color='red')

    plt.title("Total amount of monies spent on bookings")

    ax = plt.subplot(224, projection='polar')

    ax.plot(theta, data_std['price_usd'].values, 'r')
    ax.set_xticklabels(ticks)
    ax.set_xticks(np.linspace(0,2*np.pi,13))
    ax.fill_between(theta, data_std['price_usd'].values, alpha=0.2, color='red')

    plt.title("Standard Deviation")

    plt.show()
