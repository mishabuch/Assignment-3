from collections import defaultdict
from datetime import datetime
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import common_utils
from neural_network import nn_experiment

from sklearn.metrics import adjusted_mutual_info_score as ami, silhouette_score as sil_score


def cluster(cluster_range, dataset, dir):
    global start, kmeans_accuracy_for_k, end, em_prediction_y, em_pca_accuracy_for_k
    kmeans_accuracy, em_accuracy, kmeans_timetaken, em_timetaken = {}, {}, {}, {}
    sse = defaultdict(list)
    ll = defaultdict(list)
    bic = defaultdict(list)
    sil = defaultdict(lambda: defaultdict(list))
    acc = defaultdict(lambda: defaultdict(float))
    adj_mi = defaultdict(lambda: defaultdict(float))
    for k in cluster_range:
        # Kmeans Clustering
        km = KMeans(n_clusters=k, random_state=0)
        gmm = GaussianMixture(n_components=k, random_state=0)
        start = datetime.now()
        kmeans_predicted_y = km.fit_predict(dataset.x)
        end = datetime.now()

        # EM Clustering
        start = datetime.now()
        em_prediction_y = gmm.fit_predict(dataset.x)
        end = datetime.now()

        ## Accuracy

        kmeans_accuracy_for_k = common_utils.get_cluster_accuracy(dataset.y, kmeans_predicted_y)
        kmeans_accuracy[k] = kmeans_accuracy_for_k
        kmeans_timetaken[k] = (end - start).total_seconds()

        em_pca_accuracy_for_k = common_utils.get_cluster_accuracy(dataset.y, em_prediction_y)
        em_accuracy[k] = em_pca_accuracy_for_k
        em_timetaken[k] = (end - start).total_seconds()

        ## PLotting
        sil[k]['Kmeans'] = sil_score(dataset.x, kmeans_predicted_y)
        sil[k]['GMM'] = sil_score(dataset.x, em_prediction_y)

        sse[k] = [km.score(dataset.x)]
        ll[k] = [gmm.score(dataset.x)]
        bic[k] = [gmm.bic(dataset.x)]

        adj_mi[k]['Kmeans'] = ami(dataset.y, kmeans_predicted_y)
        adj_mi[k]['GMM'] = ami(dataset.y, em_prediction_y)

    sse = (-pd.DataFrame(sse)).T
    sse.index.name = 'k'
    sse.columns = ['{} sse (left)'.format(dataset.dataset_name)]

    ll = pd.DataFrame(ll).T
    ll.index.name = 'k'
    ll.columns = ['{} log-likelihood'.format(dataset.dataset_name)]

    bic = pd.DataFrame(bic).T
    bic.index.name = 'k'
    bic.columns = ['{} BIC'.format(dataset.dataset_name)]

    sil = pd.DataFrame(sil).T
    adj_mi = pd.DataFrame(adj_mi).T

    sil.index.name = 'k'
    adj_mi.index.name = 'k'

    sse.to_csv(dir + '{}_sse.csv'.format(dataset.dataset_name))
    ll.to_csv(dir + '{}_logliklihood.csv'.format(dataset.dataset_name))
    bic.to_csv(dir + '{}_bic.csv'.format(dataset.dataset_name))
    sil.to_csv(dir + '{}_sil_score.csv'.format(dataset.dataset_name))
    adj_mi.to_csv(dir + '{}_adj_mi.csv'.format(dataset.dataset_name))

    neural_net_score = nn_experiment(dataset)
    common_utils.plot_clustering_accuracy(kmeans_accuracy, "k-means - clusters vs Accuracy", dir)
    common_utils.plot_clustering_time(kmeans_timetaken, "k-means - clusters vs Time", dir)
    common_utils.plot_clustering_accuracy(em_accuracy, "EM clusters - vs Accuracy", dir)
    common_utils.plot_clustering_time(em_timetaken, "EM clusters - vs Time", dir)

    common_utils.read_and_plot_sse('Clustering', dir + '{}_sse.csv'.format(dataset.dataset_name), dir)
    common_utils.read_and_plot_loglikelihood('Clustering', dir + '{}_logliklihood.csv'.format(dataset.dataset_name),
                                             dir)
    common_utils.read_and_plot_bic('Clustering', dir + '{}_bic.csv'.format(dataset.dataset_name), dir)
    common_utils.read_and_plot_sil_score('Clustering', dir + '{}_sil_score.csv'.format(dataset.dataset_name), dir)
    common_utils.read_and_plot_adj_mi('Clustering', dir + '{}_adj_mi.csv'.format(dataset.dataset_name), dir)
