import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def correlation_matrix(df):
    # https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html#sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py
    labels = list(map(lambda x: x.split(".")[-1], df.columns.values))
    fig, ax = plt.subplots()

    matrix = df.corr(method='pearson').values

    im = ax.imshow(matrix)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

    # for i in range(len(labels)):
    #     for j in range(len(labels)):
    #         text = ax.text(j, i, matrix[i, j],
    #                        ha="center", va="center", color="w")

    fig.tight_layout()
    plt.show()

def box_plot_variable(df):
    ax = df.boxplot()

    labels = list(map(lambda x: x.split(".")[-1], df.columns.values))
    labels.insert(0, "")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
    plt.show()

def scatter_matrix_plot(df):
    # https://machinelearningmastery.com/visualize-machine-learning-data-python-pandas/
    from pandas.plotting import scatter_matrix


    to_scatter = ['activity', 'appCat.builtin', 'appCat.communication',
       'appCat.entertainment', 'appCat.finance', 'appCat.game',
       'appCat.office', 'appCat.other', 'appCat.social', 'appCat.travel',
       'appCat.unknown', 'appCat.utilities', 'appCat.weather', 'call',
       'circumplex.arousal', 'circumplex.valence', 'mood', 'screen', 'sms',]

    axes = scatter_matrix(df[to_scatter], alpha=0.2, figsize=(6, 6), diagonal='kde')

    for list in axes:
        for axe in list:
            plt.setp(axe.get_xticklabels(), visible=False)
            plt.setp(axe.get_yticklabels(), visible=False)
            axe.set_xlabel(axe.get_xlabel().split(".")[-1], rotation=45, rotation_mode='anchor', ha="right")
            axe.set_ylabel(axe.get_ylabel().split(".")[-1], rotation=0, rotation_mode='anchor', ha="right")

    plt.show()

def box_plot_id(df):

    axes = df.boxplot(by='id', column=['mood'], grid=False)
    plt.setp(axes.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
    plt.show()


def thing(df):
    df.hist()
    # plt.axis('image')
    # plt.tight_layout()
    # plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    # plt.tight_layout(pad = 1)
    # plt.rcParams.update({'font.size': 2})
    plt.show()
