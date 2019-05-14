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

    df.groupby("booking_bool").price_usd.plot.hist(logy=True, sharex=True, sharey=True, bins=20)#, range=(0,200000))
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
    """
        Pie charts do not use negative values. Therefore you should not use this after normalization of data!
    """
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

    data_sum = df.groupby(by=[pd.DatetimeIndex(df.time_of_check_in).week]).sum()
    data_describe = df.groupby(by=[pd.DatetimeIndex(df.time_of_check_in).week]).describe()

    data_sum.loc[53] = data_sum.loc[1]
    data_describe.loc[53] = data_describe.loc[1]

    data_sum = data_sum.reindex(list((v) for v in range(data_sum.index[0], data_sum.index[-1]+1))).fillna(0)
    data_describe = data_describe.reindex(list((v) for v in range(data_describe.index[0], data_describe.index[-1]+1))).fillna(0)

    # print(data_sum.columns)
    # data_sum = data_sum[data_sum['country_cluster'] == 1]
    # data_describe = data_describe[data_describe['country_cluster'] == 1]

    data_sum = data_sum['price_usd']
    data_describe = data_describe['price_usd']
    # print(data_describe.columns, data_describe['50%'])
    # print(df.country_cluster)

    theta = np.linspace(0,2*np.pi,len(data_describe.index))
    ticks = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']



    ax = plt.subplot(131, projection='polar')

    ax.plot(theta, data_describe['count'], 'r')
    ax.set_xticklabels(ticks)
    ax.set_xticks(np.linspace(0,2*np.pi,13))
    ax.fill_between(x=theta, y1=data_describe['count'],
                    alpha=0.2, color='red')

    plt.title("Number of bookings made")

    ax = plt.subplot(132, projection='polar')

    ax.plot(theta, data_describe['mean'], 'r')
    ax.set_xticklabels(ticks)
    ax.set_xticks(np.linspace(0,2*np.pi,13))
    ax.fill_between(x=theta, y1=data_describe['75%'],
                    y2=data_describe['25%'],
                    alpha=0.2, color='red')

    plt.title("Average price for bookings")

    ax = plt.subplot(133, projection='polar')

    ax.plot(theta, data_sum, 'r')
    ax.set_xticklabels(ticks)
    ax.set_xticks(np.linspace(0,2*np.pi,13))
    ax.fill_between(x=theta, y1=data_sum,
                    alpha=0.2, color='red')

    plt.title("Total sum of monies spent on bookings")

    plt.show()

def country_price_vagina_plots(df):
    sns.set(style="whitegrid")

    df_vio = df.reset_index()
    df_vio = df_vio.loc[df_vio['booking_bool'] == 1]

    # print(df_vio['visitor_location_country_id'].unique())

    filter = True
    if filter:
        countries = [219, 92, 55, 31, 220, 205, 100, 99, 130, 98, 59, 216, 129, 158, 2, 15, 132, 181]
        df_vio = df_vio.loc[df_vio['visitor_location_country_id'].isin(countries)]

    ax = sns.violinplot(y=df_vio.price_usd, x=df_vio.visitor_location_country_id,
                        data=df_vio, #palette=sns.cubehelix_palette(8),
                        hue=df_vio.random_bool,
                        split=True)

    # ax = sns.violinplot(y=df_vio.price_usd, x=df_vio.visitor_location_country_id,
    #                     data=df_vio, #palette=sns.cubehelix_palette(8),
    #                     hue=df_vio.booking_bool,
    #                     split=True)
    plt.title("Distribution of prices of booked hotels per country. Split on random bool")

    plt.show()
