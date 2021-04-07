from collections import defaultdict
from datetime import datetime

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_selection import RFE
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

import common_utils
from neural_network import nn_experiment


def clustering_rfp(cluster_range, RFE_component_diabetes, dataset, dir):
    df = dataset.data
    x = (df.iloc[:, 0:-1])
    y = (df.iloc[:, -1])
    y = y.astype('int')
    x = StandardScaler().fit_transform(x)
    global diabetes_rp, x_rp, diabetes_dataset_rp, diabetes_dataset_rp
    NN_RFE_accuracy = defaultdict(dict)
    estimator = SVR(kernel="linear")
    kmeans_accuracy_RFE = defaultdict(dict)
    kmeans_time_RFE = defaultdict(dict)
    em_accuracy_RFE = defaultdict(dict)
    em_time_RFE = defaultdict(dict)
    for RFE_comp in RFE_component_diabetes:

        diabetes_data_RFE = RFE(estimator, n_features_to_select=RFE_comp)
        diabetes_data_RFE_data = diabetes_data_RFE.fit_transform(x, y)
        diabetes_data_RFE_df = pd.DataFrame(data=diabetes_data_RFE_data)

        diabetes_rp = RFE(estimator, n_features_to_select=RFE_comp)
        x_rp = diabetes_rp.fit_transform(x, y)

        diabetes_dataset_rp = dataset
        diabetes_dataset_rp.x = x_rp
        diabetes_dataset_rp.y = y

        for cluster in cluster_range:
            # Kmean
            start = datetime.now()
            myk_mean_RFE_prediction = KMeans(n_clusters=cluster, random_state=0).fit_predict(diabetes_data_RFE_df)
            kmeans_accuracy_for_k = common_utils.get_cluster_accuracy(y, myk_mean_RFE_prediction)
            end = datetime.now()

            kmeans_accuracy_RFE[RFE_comp][cluster] = kmeans_accuracy_for_k
            kmeans_time_RFE[RFE_comp][cluster] = (end - start).total_seconds()

            # EM
            start = datetime.now()
            em_pca_prediction_y = GaussianMixture(n_components=cluster).fit(diabetes_data_RFE_df).predict(
                diabetes_data_RFE_df)
            em_pca_accuracy_for_k = common_utils.get_cluster_accuracy(y, em_pca_prediction_y)
            end = datetime.now()

            em_accuracy_RFE[RFE_comp][cluster] = em_pca_accuracy_for_k
            em_time_RFE[RFE_comp][cluster] = (end - start).total_seconds()

        NN_RFE_accuracy[RFE_comp] = nn_experiment(diabetes_dataset_rp)
    common_utils.plot_feature_transformation_time(kmeans_time_RFE, "k-means RFE clusters vs time", dir)
    common_utils.plot_feature_transformation_accuracy(kmeans_accuracy_RFE, "k-means RFE clusters vs accuracy",
                                                      dir)
    common_utils.plot_feature_transformation_time(em_time_RFE, "EM RFE clusters vs time", dir)
    common_utils.plot_feature_transformation_accuracy(em_accuracy_RFE, "EM RFE clusters vs accuracy", dir)
