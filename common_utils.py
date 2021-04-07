from sklearn.metrics import accuracy_score
from collections import Counter

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from kneed import KneeLocator

from numpy import matlib as mb


def get_cluster_accuracy(data_y, prediction_y):
    final_prediction_labels = np.empty_like(data_y)
    for i in set(prediction_y):
        mask = prediction_y == i
        target = Counter(data_y[mask]).most_common(1)[0][0]
        final_prediction_labels[mask] = target
    return accuracy_score(data_y, final_prediction_labels)


def plot_clustering_accuracy(dataset, title, mydir):
    fig = plt.figure()

    n_clusters = list(dataset.keys())
    accuracy = list(dataset.values())

    ax = fig.add_subplot(111, xlabel='Number of Clusters', ylabel='Accuracy', title=title)

    ax.plot(n_clusters, accuracy, 'o-', color="b",
            label="Accuracy")

    ax.set_xticks(n_clusters)

    ax.legend(loc="best")
    fig.savefig(mydir + title + ".png")  # save the figure to file
    plt.close(fig)
    return plt


def plot_clustering_time(datadictionary, mytitle, mydir):
    fig = plt.figure()

    n_clusters = list(datadictionary.keys())
    time = list(datadictionary.values())

    ax = fig.add_subplot(111, xlabel='Number of Clusters', ylabel='Time', title=mytitle)

    ax.plot(n_clusters, time, 'o-', color="b",
            label="Number of Clusters")
    ax.set_xticks(n_clusters)

    ax.legend(loc="best")
    fig.savefig(mydir + mytitle + ".png")
    plt.close(fig)
    return plt


def plot_feature_transformation_accuracy(datadictionary, mytitle, mydir):
    fig = plt.figure()

    PCA_components = list(datadictionary.keys())

    ax = fig.add_subplot(111, xlabel='Number of Clusters', ylabel='Accuracy', title=mytitle)

    for PCA_comp in range(len(PCA_components)):
        n_clusters = list(datadictionary[PCA_components[PCA_comp]].keys())
        accuracy = list(datadictionary[PCA_components[PCA_comp]].values())
        ax.plot(n_clusters, accuracy, 'o-',
                label=str(PCA_components[PCA_comp]))

    ax.set_xticks(n_clusters)

    ax.legend(loc="best")
    fig.savefig(mydir + mytitle + ".png")
    plt.close(fig)
    return plt


def plot_feature_transformation_time(datadictionary, mytitle, mydir):
    fig = plt.figure()

    PCA_components = list(datadictionary.keys())

    ax = fig.add_subplot(111, xlabel='Number of Clusters', ylabel='Accuracy', title=mytitle)

    for PCA_comp in range(len(PCA_components)):
        n_clusters = list(datadictionary[PCA_components[PCA_comp]].keys())
        time = list(datadictionary[PCA_components[PCA_comp]].values())
        ax.plot(n_clusters, time, 'o-',
                label=str(PCA_components[PCA_comp]))

    ax.set_xticks(n_clusters)

    ax.legend(loc="best")
    fig.savefig(mydir + mytitle + ".png")
    plt.close(fig)
    return plt


def plot_kmeans_gmm(title, df, xlabel='Number of Clusters', ylabel='Accuracy'):
    plt.close()
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.tight_layout()

    plt.plot(df.index.values, df['Kmeans'], 'o-', linewidth=1, markersize=2,
             label="k-Means")
    plt.plot(df.index.values, df['GMM'], 'o-', linewidth=1, markersize=2,
             label="GMM")
    plt.legend(loc="best")

    return plt


def plot_sse(title, df, xlabel='Number of Clusters', ylabel='SSE'):
    plt.close()
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.tight_layout()

    plt.plot(df.index.values, df.iloc[:, 0], 'o-', linewidth=1, markersize=2)
    plt.legend(loc="best")

    return plt


def plot_loglikelihood(title, df, xlabel='Number of Clusters', ylabel='Log Likelihood'):
    plt.close()
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.tight_layout()

    plt.plot(df.index.values, df.iloc[:, 0], 'o-', linewidth=1, markersize=2)
    plt.legend(loc="best")

    return plt


def plot_bic(title, df, xlabel='Number of Clusters', ylabel='BIC'):
    plt.close()
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.tight_layout()

    plt.plot(df.index.values, df.iloc[:, 0], 'o-', linewidth=1, markersize=2)
    plt.legend(loc="best")

    return plt


def plot_adj_mi(title, df):
    return plot_kmeans_gmm(title, df, ylabel='Adj. MI')


def plot_sil_score(title, df):
    return plot_kmeans_gmm(title, df, 'Number of Clusters', 'Silhouette Score')


def read_and_plot_sse(problem, file, output_dir):
    title = '{} : SSE vs Number of Clusters'.format(problem)
    df = pd.read_csv(file).set_index('k')
    p = plot_sse(title, df)
    p.savefig(
        '{}{}_sse.png'.format(output_dir, problem),
        format='png', bbox_inches='tight', dpi=150)


# Reference - https://dataplatform.cloud.ibm.com/analytics/notebooks/54d79c2a-f155-40ec-93ec-ed05b58afa39/view
# ?access_token=6d8ec910cf2a1b3901c721fcb94638563cd646fe14400fecbb76cea6aaae2fb1
def find_knee(values):
    # get coordinates of all the points
    nPoints = len(values)
    allCoord = np.vstack((range(nPoints), values)).T
    # np.array([range(nPoints), values])

    # get the first point
    firstPoint = allCoord[0]
    # get vector between first and last point - this is the line
    lineVec = allCoord[-1] - allCoord[0]
    lineVecNorm = lineVec / np.sqrt(np.sum(lineVec ** 2))

    # find the distance from each point to the line:
    # vector between all points and first point
    vecFromFirst = allCoord - firstPoint

    # To calculate the distance to the line, we split vecFromFirst into two
    # components, one that is parallel to the line and one that is perpendicular
    # Then, we take the norm of the part that is perpendicular to the line and
    # get the distance.
    # We find the vector parallel to the line by projecting vecFromFirst onto
    # the line. The perpendicular vector is vecFromFirst - vecFromFirstParallel
    # We project vecFromFirst by taking the scalar product of the vector with
    # the unit vector that points in the direction of the line (this gives us
    # the length of the projection of vecFromFirst onto the line). If we
    # multiply the scalar product by the unit vector, we have vecFromFirstParallel
    scalarProduct = np.sum(vecFromFirst * mb.repmat(lineVecNorm, nPoints, 1), axis=1)
    vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
    vecToLine = vecFromFirst - vecFromFirstParallel

    # distance to line is the norm of vecToLine
    distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))

    # knee/elbow is the point with max distance value
    idxOfBestPoint = np.argmax(distToLine)
    return idxOfBestPoint


def plot_dim_red_scores(file, dir, dataset_name, problem_name, multiple_runs=False, xlabel='Number of Clusters',
                        ylabel=None):
    title = '{} - {}: '.format(dataset_name, problem_name) + '{} vs Number of Components'
    df = pd.read_csv(file, header=None).dropna().set_index(0)
    if ylabel is None:
        ylabel = 'Kurtosis'
        if problem_name == 'PCA' or problem_name == 'SVD':
            ylabel = 'Variance'
        elif problem_name == 'RP':
            # ylabel = 'PDCC'  # 'Pairwise distance corrcoef'
            ylabel = 'Pairwise distance corrcoef'
        elif problem_name == 'RFE':
            ylabel = 'Feature Importances'
    title = title.format(ylabel)

    plt.close()
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.tight_layout()

    ax = plt.gca()

    x_points = df.index.values
    y_points = df[1]
    if multiple_runs:
        y_points = np.mean(df.iloc[:, 1:-1], axis=1)
        y_std = np.std(df.iloc[:, 1:-1], axis=1)
        plt.plot(x_points, y_points, 'o-', linewidth=1, markersize=2,
                 label=ylabel)
        plt.fill_between(x_points, y_points - y_std,
                         y_points + y_std, alpha=0.2)
    else:
        plt.plot(x_points, y_points, 'o-', linewidth=1, markersize=2,
                 label=ylabel)

    min_value = np.min(y_points)
    min_point = y_points.idxmin()
    max_value = np.max(y_points)
    max_point = y_points.idxmax()
    knee_point = find_knee(y_points)
    kl = KneeLocator(x_points, y_points)

    ax.axvline(x=min_point, linestyle="--", label="Min: {}".format(int(min_point)))
    ax.axvline(x=max_point, linestyle="--", label="Max: {}".format(int(max_point)))

    ax.axvline(x=knee_point, linestyle="--", label="Knee: {}".format(knee_point))
    ax.set_xticks(df.index.values, minor=False)

    plt.legend(loc="best")

    plt.savefig(
        '{}{}_{}_scree.png'.format(dir, problem_name, dataset_name),
        format='png', bbox_inches='tight', dpi=150)

    return plt


def read_and_plot_adj_mi(problem, file, output_dir):
    title = '{} : Adj. MI vs Number of Clusters'.format(problem)
    df = pd.read_csv(file).set_index('k')
    p = plot_adj_mi(title, df)
    p.savefig(
        '{}{}_adjMI.png'.format(output_dir, problem),
        format='png', bbox_inches='tight', dpi=150)


def read_and_plot_loglikelihood(problem, file, output_dir):
    title = '{} : Log Likelihood vs Number of Clusters'.format(problem)
    df = pd.read_csv(file).set_index('k')
    p = plot_loglikelihood(title, df)
    p.savefig(
        '{}{}_loglikelihood.png'.format(output_dir, problem),
        format='png', bbox_inches='tight', dpi=150)


def read_and_plot_bic(problem, file, output_dir):
    title = '{} : BIC vs Number of Clusters'.format(problem)
    df = pd.read_csv(file).set_index('k')
    p = plot_bic(title, df)
    p.savefig(
        '{}{}_bic.png'.format(output_dir, problem),
        format='png', bbox_inches='tight', dpi=150)


def read_and_plot_sil_score(problem, file, output_dir):
    title = '{} : Silhouette Score vs Number of Clusters'.format(problem)
    df = pd.read_csv(file).set_index('k')
    p = plot_sil_score(title, df)
    p.savefig(
        '{}{}_sil_score.png'.format(output_dir, problem),
        format='png', bbox_inches='tight', dpi=150)
