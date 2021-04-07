from collections import defaultdict
from datetime import datetime
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import common_utils
from neural_network import nn_experiment


def clustering_pca(cluster_range, _pca_components_range, dataset, dir):
    global _pca, df, x, y, x_pca, _dataset_pca, _dataset_pca
    NN_PCA_accuracy = defaultdict(dict)
    # for  PCA, we can only have 20 Principal components since the number of features for  is 20
    kmeans_accuracy_pca, kmeans_time_pca, _accuracy_em_PCA, _time_em_PCA = defaultdict(
        dict), defaultdict(
        dict), defaultdict(dict), defaultdict(dict)
    _pca = PCA(random_state=0)
    eigen = _pca.fit(dataset.x)
    tmp = pd.Series(data=_pca.explained_variance_, index=range(1, min(_pca.explained_variance_.shape[0], 500) + 1))
    tmp.to_csv(dir+'{}_pca_scree.csv'.format(dataset.dataset_name))
    common_utils.plot_dim_red_scores(dir + '{}_pca_scree.csv'.format(dataset.dataset_name), dir, dataset.dataset_name, "PCA", multiple_runs=False, xlabel='Number of Clusters', ylabel=None)
    df = dataset.data
    x = (df.iloc[:, 0:-1])
    y = (df.iloc[:, -1])
    y = y.astype('int')
    x = StandardScaler().fit_transform(x)
    for component in _pca_components_range:

        _pca = PCA(n_components=component, random_state=0)
        x_pca = _pca.fit_transform(x)

        _dataset_pca = dataset
        _dataset_pca.x = x_pca
        _dataset_pca.y = y

        for cluster in cluster_range:
            # Kmeans
            start = datetime.now()
            kmeans_pca_prediction_y = KMeans(n_clusters=cluster, random_state=0).fit_predict(_dataset_pca.x)
            kmeans_accuracy_for_k = common_utils.get_cluster_accuracy(_dataset_pca.y, kmeans_pca_prediction_y)
            end = datetime.now()

            kmeans_accuracy_pca[component][cluster] = kmeans_accuracy_for_k
            kmeans_time_pca[component][cluster] = (end - start).total_seconds()

            # EM
            start = datetime.now()
            em_pca_prediction_y = GaussianMixture(n_components=cluster).fit_predict(_dataset_pca.x)
            em_pca_accuracy_for_k = common_utils.get_cluster_accuracy(_dataset_pca.y, em_pca_prediction_y)
            end = datetime.now()

            _accuracy_em_PCA[component][cluster] = em_pca_accuracy_for_k
            _time_em_PCA[component][cluster] = (end - start).total_seconds()

        NN_PCA_accuracy[component] = nn_experiment(_dataset_pca)
    common_utils.plot_feature_transformation_time(kmeans_time_pca, "k-means PCA clusters vs time", dir)
    common_utils.plot_feature_transformation_accuracy(kmeans_accuracy_pca, "k-means PCA clusters vs accuracy", dir)
    common_utils.plot_feature_transformation_time(_time_em_PCA, "EM PCA clusters vs time", dir)
    common_utils.plot_feature_transformation_accuracy(_accuracy_em_PCA, "EM PCA clusters vs accuracy", dir)