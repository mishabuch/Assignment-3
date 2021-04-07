import os
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection
from sklearn.svm import SVR
from datetime import datetime
import dataset_loader
from clustering import cluster
from ica import clustering_ica
from neural_network import nn_experiment
from pca import clustering_pca
from rfp import clustering_rfp
from rp import clustering_rp
import pandas as pd
assignment3_graphs = r"/Users/amisha/Documents/Gatech/Assignment 3/diabetes/"
os.chdir(assignment3_graphs)

if __name__ == '__main__':
    # load diabetes data - it is transformed and standardized already
    datasets = dataset_loader.load_datasets()
    diabetes_dataset = datasets[0]
    cluster_range = [2, 3, 4, 5, 6, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    df = diabetes_dataset.data
    x = (df.iloc[:, 0:-1])
    y = (df.iloc[:, -1])
    y = y.astype('int')
    x = StandardScaler().fit_transform(x)

    # Kmeans and EM  clustering
    cluster(cluster_range, diabetes_dataset, assignment3_graphs)

    # PCA + Kmeans and EM
    diabetes_pca_components_range = [1, 2, 5, 8, 10, 14, 16, 18, 19]
    clustering_pca(cluster_range, diabetes_pca_components_range, diabetes_dataset, assignment3_graphs)

    # ICA + Kmeans and EM
    diabetes_ica_components_range = [1, 2, 5, 8, 10, 14, 16, 18, 19]
    clustering_ica(cluster_range, diabetes_ica_components_range, diabetes_dataset, assignment3_graphs)

    # RP + Kmeans and EM
    # for diabetes ICA, we can only have 11 Principal components since the number of features for diabetes is 11
    diabetes_RP_components_range = [1, 2, 5, 8, 10, 14, 16, 18, 19]
    clustering_rp(cluster_range, diabetes_RP_components_range, diabetes_dataset, assignment3_graphs)

    # RFE + Kmeans and EM
    # for diabetes ICA, we can only have 11 Principal components since the number of features for diabetes is 11
    diabetes_RFE_components_range = [1, 2, 5, 8, 10, 14, 16, 18, 19]
    clustering_rfp(cluster_range, diabetes_RP_components_range, diabetes_dataset, assignment3_graphs)

    ''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # NN with clustering and dimensionality reduction
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''

    # PCA Kmeans
    start = datetime.now()
    diabetes_pca = PCA(n_components=1, random_state=0)
    x_pca = diabetes_pca.fit_transform(x)

    diabetes_dataset_pca = diabetes_dataset
    diabetes_dataset_pca.x = x_pca
    diabetes_dataset_pca.y = y

    kmeans_pca_prediction_y = KMeans(n_clusters=50, random_state=0).fit_predict(diabetes_dataset_pca.x)

    diabetes_dataset_pca_kmean_nn = diabetes_dataset
    diabetes_dataset_pca_kmean_nn.x = x_pca
    diabetes_dataset_pca.y = kmeans_pca_prediction_y

    NN_PCA_NN_kmean_score = nn_experiment(diabetes_dataset_pca)
    end = datetime.now()

    NN_PCA_NN_kmean_time = (end - start).total_seconds()

    # PCA EM
    start = datetime.now()

    em_pca_prediction_y = GaussianMixture(n_components=50).fit_predict(diabetes_dataset_pca.x)
    diabetes_dataset_pca.y = em_pca_prediction_y

    NN_PCA_NN_em_score = nn_experiment(diabetes_dataset_pca)
    end = datetime.now()

    NN_PCA_NN_em_time = (end - start).total_seconds()

    # ICA Kmeans
    start = datetime.now()
    diabetes_ica = FastICA(n_components=16, random_state=0)
    x_ica = diabetes_ica.fit_transform(x)

    diabetes_dataset_ica = diabetes_dataset
    diabetes_dataset_ica.x = x_ica
    diabetes_dataset_ica.y = y

    myk_mean_ICA_prediction = KMeans(n_clusters=50, random_state=0).fit_predict(diabetes_dataset_ica.x)

    diabetes_dataset_ica_kmean_nn = diabetes_dataset
    diabetes_dataset_ica_kmean_nn.x = x_ica
    diabetes_dataset_ica.y = myk_mean_ICA_prediction

    NN_ICA_NN_kmean_score = nn_experiment(diabetes_dataset_ica)
    end = datetime.now()

    NN_ICA_NN_kmean_time = (end - start).total_seconds()

    # ICA em
    start = datetime.now()
    myk_em_ICA_prediction = GaussianMixture(n_components=50).fit_predict(diabetes_dataset_ica.x)

    diabetes_dataset_ica.y = myk_em_ICA_prediction

    NN_ICA_NN_EM_score = nn_experiment(diabetes_dataset_ica)
    end = datetime.now()

    NN_ICA_NN_EM_time = (end - start).total_seconds()

    # RP Kmeans
    start = datetime.now()
    diabetes_rp = GaussianRandomProjection(n_components=50, random_state=0)
    x_rp = diabetes_rp.fit_transform(x)

    diabetes_dataset_rp = diabetes_dataset
    diabetes_dataset_rp.x = x_rp
    diabetes_dataset_rp.y = y

    k_mean_RP_NN_prediction = KMeans(n_clusters=50, random_state=0).fit_predict(diabetes_dataset_rp.x)
    diabetes_dataset_rp.y = k_mean_RP_NN_prediction
    NN_RP_NN_kmean_score = nn_experiment(diabetes_dataset_rp)
    end = datetime.now()

    NN_RP_NN_kmean_time = (end - start).total_seconds()

    # RP EM
    start = datetime.now()
    myk_em_RP_NN_prediction = GaussianMixture(n_components=50).fit_predict(diabetes_dataset_rp.x)
    diabetes_dataset_rp.y = myk_em_RP_NN_prediction
    NN_RP_NN_EM_score = nn_experiment(diabetes_dataset_rp)
    end = datetime.now()

    NN_RP_NN_EM_time = (end - start).total_seconds()
