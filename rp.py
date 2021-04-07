from collections import defaultdict
from datetime import datetime
import scipy.sparse as sps
import pandas as pd
import numpy as np
from scipy.linalg import pinv
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection
from itertools import product
import common_utils
from neural_network import nn_experiment


def pairwise_dist_corr(x1, x2):
    assert x1.shape[0] == x2.shape[0]

    d1 = pairwise_distances(x1)
    d2 = pairwise_distances(x2)
    return np.corrcoef(d1.ravel(), d2.ravel())[0, 1]


def reconstruction_error(projections, x):
    w = projections.components_
    if sps.issparse(w):
        w = w.todense()
    p = pinv(w)
    reconstructed = ((p @ w) @ (x.T)).T  # Unproject projected data
    errors = np.square(x - reconstructed)
    return np.nanmean(errors)


def clustering_rp(cluster_range, RP_component, dataset, dir):
    df = dataset.data
    x = (df.iloc[:, 0:-1])
    y = (df.iloc[:, -1])
    y = y.astype('int')
    x = StandardScaler().fit_transform(x)
    global diabetes_rp, x_rp, diabetes_dataset_rp, diabetes_dataset_rp
    NN_RP_ = defaultdict(dict)
    kmeans_accuracy_RP = defaultdict(dict)
    kmeans_time_RP = defaultdict(dict)
    em_accuracy_RP = defaultdict(dict)
    em_time_RP = defaultdict(dict)

    tmp = defaultdict(dict)
    for i, dim in product(range(10), RP_component):
        rp = GaussianRandomProjection(random_state=i, n_components=dim)
        tmp[dim][i] = pairwise_dist_corr(rp.fit_transform(dataset.x), dataset.x)
    tmp = pd.DataFrame(tmp).T
    tmp.to_csv(dir + '{}_rp_scree1.csv'.format(dataset.dataset_name))
    common_utils.plot_dim_red_scores(dir + '{}_rp_scree1.csv'.format(dataset.dataset_name), dir, dataset.dataset_name,
                                     "RP", multiple_runs=False, xlabel='Number of Clusters', ylabel=None)

    tmp = defaultdict(dict)
    for i, dim in product(range(10), RP_component):
        rp = GaussianRandomProjection(random_state=i, n_components=dim)
        rp.fit(dataset.x)
        tmp[dim][i] = reconstruction_error(rp, dataset.x)
    tmp = pd.DataFrame(tmp).T
    tmp.to_csv(dir + '{}_rp_scree2.csv'.format(dataset.dataset_name))
    common_utils.plot_dim_red_scores(dir + '{}_rp_scree2.csv'.format(dataset.dataset_name), dir, dataset.dataset_name,
                                     "RP", multiple_runs=False, xlabel='Number of Clusters', ylabel=None)

    for RP_comp in RP_component:

        diabetes_data_RP = GaussianRandomProjection(n_components=RP_comp, random_state=0)
        diabetes_data_RP_data = diabetes_data_RP.fit_transform(x)
        diabetes_data_RP_df = pd.DataFrame(data=diabetes_data_RP_data)

        diabetes_rp = GaussianRandomProjection(n_components=RP_comp, random_state=0)
        x_rp = diabetes_rp.fit_transform(x)

        diabetes_dataset_rp = dataset
        diabetes_dataset_rp.x = x_rp
        diabetes_dataset_rp.y = y

        for cluster in cluster_range:
            # Kmeans
            start = datetime.now()
            myk_mean_RP_prediction = KMeans(n_clusters=cluster, random_state=0).fit_predict(diabetes_data_RP_df)
            kmeans_accuracy_for_k = common_utils.get_cluster_accuracy(y, myk_mean_RP_prediction)
            end = datetime.now()

            kmeans_accuracy_RP[RP_comp][cluster] = kmeans_accuracy_for_k
            kmeans_time_RP[RP_comp][cluster] = (end - start).total_seconds()

            # EM
            start = datetime.now()
            em_pca_prediction_y = GaussianMixture(n_components=cluster).fit(diabetes_data_RP_df).predict(
                diabetes_data_RP_df)
            em_pca_accuracy_for_k = common_utils.get_cluster_accuracy(y, em_pca_prediction_y)
            end = datetime.now()

            em_accuracy_RP[RP_comp][cluster] = em_pca_accuracy_for_k
            em_time_RP[RP_comp][cluster] = (end - start).total_seconds()

        NN_RP_[RP_comp] = nn_experiment(diabetes_dataset_rp)
    common_utils.plot_feature_transformation_time(kmeans_time_RP, "k-means RP clusters vs time", dir)
    common_utils.plot_feature_transformation_accuracy(kmeans_accuracy_RP, "k-means RP clusters vs score", dir)
    common_utils.plot_feature_transformation_time(em_time_RP, "EM RP clusters vs time", dir)
    common_utils.plot_feature_transformation_accuracy(em_accuracy_RP, "EM RP clusters vs score", dir)
