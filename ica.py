from collections import defaultdict
from datetime import datetime

from sklearn.cluster import KMeans
from sklearn.decomposition import FastICA
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import common_utils
from neural_network import nn_experiment


def clustering_ica(cluster_range, ICA_component_, dataset, dir):
    df = dataset.data
    x = (df.iloc[:, 0:-1])
    y = (df.iloc[:, -1])
    y = y.astype('int')
    x = StandardScaler().fit_transform(x)
    global _ica, x_ica, _dataset_ica, _dataset_ica
    NN_ICA_accuracy = defaultdict(dict)
    kmeans_accuracy_ICA = defaultdict(dict)
    kmeans_time_ICA = defaultdict(dict)
    em_accuracy_ICA = defaultdict(dict)
    em_time_ICA = defaultdict(dict)
    _data_ICA = FastICA(random_state=0)
    kurt = {}
    for dim in ICA_component_:
        _data_ICA.set_params(n_components=dim)
        tmp = _data_ICA.fit_transform(dataset.x)
        tmp = pd.DataFrame(tmp)
        tmp = tmp.kurt(axis=0)
        kurt[dim] = tmp.abs().mean()

    kurt = pd.Series(kurt)
    kurt.to_csv(dir + '{}_ica_scree.csv'.format(dataset.dataset_name))
    common_utils.plot_dim_red_scores(dir + '{}_ica_scree.csv'.format(dataset.dataset_name), dir, dataset.dataset_name,
                                     "ICA", multiple_runs=False, xlabel='Number of Clusters', ylabel=None)

    _data_ICA_data = _data_ICA.fit_transform(x)
    _data_ICA_df = pd.DataFrame(data=_data_ICA_data)
    _data_ICA_kurtosis = _data_ICA_df.kurt()
    print(_data_ICA_kurtosis)
    for ICA_comp in ICA_component_:

        _data_ICA = FastICA(n_components=ICA_comp, random_state=0)
        _data_ICA_data = _data_ICA.fit_transform(x)
        _data_ICA_df = pd.DataFrame(data=_data_ICA_data)

        _ica = FastICA(n_components=ICA_comp, random_state=0)
        x_ica = _ica.fit_transform(x)

        _dataset_ica = dataset
        _dataset_ica.x = x_ica
        _dataset_ica.y = y

        for cluster in cluster_range:
            # Kmeans
            start = datetime.now()
            myk_mean_ICA_prediction = KMeans(n_clusters=cluster, random_state=0).fit_predict(_data_ICA_df)
            kmeans_accuracy_for_k = common_utils.get_cluster_accuracy(y, myk_mean_ICA_prediction)
            end = datetime.now()

            kmeans_accuracy_ICA[ICA_comp][cluster] = kmeans_accuracy_for_k
            kmeans_time_ICA[ICA_comp][cluster] = (end - start).total_seconds()

            # EM
            start = datetime.now()
            em_pca_prediction_y = GaussianMixture(n_components=cluster).fit(_data_ICA_df).predict(
                _data_ICA_df)
            em_pca_accuracy_for_k = common_utils.get_cluster_accuracy(y, em_pca_prediction_y)
            end = datetime.now()

            em_accuracy_ICA[ICA_comp][cluster] = em_pca_accuracy_for_k
            em_time_ICA[ICA_comp][cluster] = (end - start).total_seconds()

        NN_ICA_accuracy[ICA_comp] = nn_experiment(_dataset_ica)
    common_utils.plot_feature_transformation_time(kmeans_time_ICA, "k-means ICA clusters vs time", dir)
    common_utils.plot_feature_transformation_accuracy(kmeans_accuracy_ICA, "k-means ICA clusters vs accuracy",
                                                      dir)
    common_utils.plot_feature_transformation_time(em_time_ICA, "EM ICA clusters vs time", dir)
    common_utils.plot_feature_transformation_accuracy(em_accuracy_ICA, "EM ICA clusters vs accuracy", dir)
