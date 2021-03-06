import pandas as pd
from datetime import datetime
from datetime import timedelta
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from math import radians
import csv

def histogram_site_id (df):
    plt.hist(df['site_id'], bins=len(df['site_id'].unique()), log=True)
    plt.show()

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

    ax = scatter_data.plot(kind='scatter', x='mean', y='std', c='country_cluster', colormap='jet')
    plt.show()

def box_plot_variable(df):
    sns.set(style="whitegrid")
    to_drop = ['prop_id', 'site_id', 'visitor_location_country_id', 'prop_country_id', 'orig_destination_distance',
                'srch_destination_id', 'srch_saturday_night_bool', 'click_bool', 'booking_bool', 'random_bool',
               'promotion_flag', 'prop_brand_bool',
               'price_usd', 'comp_diff_avg', 'gross_bookings_usd', 'visitor_hist_adr_usd',
               'srch_booking_window', 'srch_query_affinity_score']


    df_boxplot = df.drop(columns=to_drop)
    # ax = sns.boxplot(data=df_boxplot)
    ax = df_boxplot.boxplot(showfliers=True, )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.show()

    to_loop = ['comp_diff_avg', 'gross_bookings_usd', 'visitor_hist_adr_usd',
                        'srch_booking_window', 'srch_query_affinity_score']
    for variable in to_loop:

        ax = df.boxplot(column=variable, showfliers=True, )
        # ax = sns.boxplot(data=df_boxplot, )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
        plt.gcf().subplots_adjust(bottom=0.2)
        plt.show()

def show_me_the_money(df, range=2000):

    df.groupby("booking_bool").price_usd.plot.hist(logy=True, sharex=True, sharey=True, bins=20, range=(0, range))
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
    # df = df[df['country_cluster'] == 2]

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

    sorter = df_vio.groupby('visitor_location_country_id').mean()
    # print(sorter)
    sorter = sorter.sort_values("price_usd").index
    ax = sns.violinplot(y=df_vio.price_usd, x=df_vio.visitor_location_country_id,
                        data=df_vio, #palette=sns.cubehelix_palette(8),
                        hue=df_vio.random_bool,
                        split=True,
                        order = sorter)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
    # ax = sns.violinplot(y=df_vio.price_usd, x=df_vio.visitor_location_country_id,
    #                     data=df_vio, #palette=sns.cubehelix_palette(8),
    #                     hue=df_vio.booking_bool,
    #                     split=True)
    plt.title("Distribution of prices of booked hotels per country. Split on random bool")

    plt.show()

def position_bias(df):
    df_1 = df[df['random_bool'] == 1]
    df_2 = df[df['random_bool'] == 0]

    print(df_1.columns)
    df_1.plot.bar(x=df_1['hotel_position_mean'] , y=df_1.count())
    plt.show()

def visualize_trainings():
    path = '../models/training_curves.csv'

    iterations = []
    train_scores = []
    Cs = []
    Bs = []
    Ss = []

    with open(path, 'r', newline='\n') as f:
        reader = csv.reader(f, delimiter=';')
        for i, row in enumerate(reader):
            if i == 0:
                continue
            # print(row)
            iterations.append(int(row[0]))
            train_scores.append(float(row[1]))
            monitor_output = row[-1].replace(" ", "").split(":")
            monitor_output.pop(0)
            Cs.append(float(monitor_output[0][:-1]))
            Bs.append(float(monitor_output[1][:-1]))
            Ss.append(int(monitor_output[-1]))

    plt.plot(iterations, train_scores, label='Training score')
    plt.plot(iterations, Cs, label='Current Validation score')
    plt.axhline(Bs[-1], color='red', linestyle='--', label='Best Validation score')
    # plt.plot(iterations, , label='Best Validation score')
    plt.legend()
    plt.title("Training results LambdaMART")
    plt.xlabel("Iteration")
    plt.show()

def booking_days(df):

    df = df[df['booking_bool'] == 1]
    booking_doys = df['date_time'].apply(lambda x: x.timetuple().tm_yday)
    # booking_doys.plot.hist(bins=365, logy=True, alpha=0.5, label="Booking date")
    sns.distplot(booking_doys, bins=365, label="Date when booking made", kde=False)


    booking_doys = df['time_of_check_in'].apply(lambda x: x.timetuple().tm_yday)
    # booking_doys.plot.hist(bins=365, logy=True, alpha=0.5, label="Check in date")
    sns.distplot(booking_doys, bins=365, label="Check in date", kde=False)

    plt.xlim(0, 365)
    plt.title("Check in times and booking dates of the dataset")
    plt.legend()
    plt.show()

def polar_booking_days(df):

    df = df[df['booking_bool'] == 1]
    df = df[['price_usd', 'date_time', 'time_of_check_in']]
    # df = df[df['country_cluster'] == 2]

    data_date_time = df.groupby(by=[pd.DatetimeIndex(df.date_time).dayofyear]).describe()
    data_time_of_check_in = df.groupby(by=[pd.DatetimeIndex(df.time_of_check_in).dayofyear]).describe()

    data_date_time.loc[366] = data_date_time.loc[1]
    data_time_of_check_in.loc[366] = data_time_of_check_in.loc[1]

    data_date_time = data_date_time.reindex(list((v) for v in range(data_date_time.index[0], data_date_time.index[-1]+1))).fillna(1)
    data_time_of_check_in = data_time_of_check_in.reindex(list((v) for v in range(data_time_of_check_in.index[0], data_time_of_check_in.index[-1]+1))).fillna(1)

    # print(data_date_time['price_usd']['count'], len(data_date_time.index))
    # data_date_time['price_usd']['count'].plot()
    # data_time_of_check_in['price_usd']['count'].plot()
    # plt.show()

    data_date_time = data_date_time['price_usd']
    data_time_of_check_in = data_time_of_check_in['price_usd']

    theta = np.linspace(0, 2*np.pi, len(data_date_time.index))
    ticks = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
    width = np.pi / 300
    ax = plt.subplot(111, projection='polar')

    ax.bar(theta, data_date_time['count'], width=width, label="date of booking",  alpha=1)
    ax.set_xticklabels(ticks)
    ax.set_xticks(np.linspace(0,2*np.pi,13))


    ax.bar(theta, data_time_of_check_in['count'], width=width, label="time of check in", alpha=1)
    ax.set_xticklabels(ticks)
    ax.set_xticks(np.linspace(0,2*np.pi,13))


    plt.title("Number of bookings made")
    plt.legend()
    plt.show()

def children_taken(df):
    df = df[df['booking_bool'] == 1]
    df['Days'] = df['date_time'].apply(lambda x: x.timetuple().tm_yday)
    # print(df.groupby('temp').mean().srch_adults_count)
    to_plot = df.groupby('Days').sum().srch_adults_count
    # booking_doys = df['srch_adults_count'].apply(lambda x: x.timetuple().tm_yday)
    to_plot.plot(alpha=0.5, label="Adults mean")
    # sns.distplot(to_plot, bins=365, label="Adults count", kde=False)

    to_plot2 = df.groupby('Days').sum().srch_children_count
    uuh_hoe_doeikdit = to_plot / to_plot2
    # print(df.groupby('temp').mean().srch_children_count)
    to_plot2.plot(alpha=0.5, label="Children mean")
    uuh_hoe_doeikdit.plot(alpha=0.5, label="Ratio")
    # sns.distplot(to_plot, bins=365, label="Children count", kde=False)

    plt.xlim(0, 365)
    plt.title("Children and adult plot")
    plt.legend()
    plt.show()

def length_of_stay(df):
    df = df[df['booking_bool'] == 1]
    df['Days'] = df['date_time'].apply(lambda x: x.timetuple().tm_yday)

    to_plot = df.groupby('Days').mean().srch_length_of_stay

    to_plot.plot(alpha=0.5, label="length of stay")

    plt.xlim(0, 365)
    plt.title("Length of stays")
    plt.legend()
    plt.show()

def booking_window(df):
    df = df[df['booking_bool'] == 1]

    df_to_use = df[df['orig_destination_distance'] != 0]
    # sns.regplot(x="srch_booking_window", y="srch_children_count", data=df)
    # sns.regplot(x="srch_booking_window", y="price_usd", data=df)
    # ax = sns.scatterplot(x="srch_booking_window", y="orig_destination_distance", data=df_to_use, alpha=0.3)
    # sns.relplot(x="srch_booking_window", y="orig_destination_distance", kind="line", data=df_to_use, ax=ax)

    sns.lmplot(x="srch_booking_window", y="orig_destination_distance", data=df_to_use,
               order=1, ci=None, scatter_kws={"s": 80})

    # sns.lmplot(x="srch_booking_window", y="srch_length_of_stay", data=df_to_use,
    #            order=1, ci=None, scatter_kws={"s": 80})
    # g.get_axes()[0].set_yscale('log')
    # g.set_yscale('log')
    plt.ylim(0, None)
    plt.title("Booking window plots")
    plt.legend()
    plt.show()

def scatterplot_stuff(df):
    df = df[df['booking_bool'] == 1]
    df.orig_destination_distance.plot.density()
    plt.show()
if __name__ == "__main__":
    visualize_trainings()
