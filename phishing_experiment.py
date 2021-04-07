import os
from sklearn.preprocessing import StandardScaler
import dataset_loader
from clustering import cluster
from ica import clustering_ica
from pca import clustering_pca
from rfp import clustering_rfp
from rp import clustering_rp

assignment3_graphs = r"/Users/amisha/Documents/Gatech/Assignment 3/phishing/"
os.chdir(assignment3_graphs)

if __name__ == '__main__':
    # load phishing data - it is transformed and standardized already
    datasets = dataset_loader.load_datasets()
    phishing_dataset = datasets[1]
    cluster_range = [2, 3, 4, 5, 6, 7, 10, 15, 20, 25, 30]
    df = phishing_dataset.data
    x = (df.iloc[:, 0:-1])
    y = (df.iloc[:, -1])
    y = y.astype('int')
    x = StandardScaler().fit_transform(x)

    # Kmeans and EM clustering
    cluster(cluster_range, phishing_dataset, assignment3_graphs)

    # PCA + Kmeans and EM
    PCA_component_phishing = [1, 2, 5, 8, 9]
    clustering_pca(cluster_range,PCA_component_phishing,phishing_dataset,assignment3_graphs)

    # ICA + Kmeans and EM
    ICA_component_phishing = [1, 2, 5, 8, 9]
    clustering_ica(cluster_range,ICA_component_phishing,phishing_dataset,assignment3_graphs)

    # RP + Kmeans and EM
    # for phishing ICA, we can only have 11 Principal components since the number of features for phishing is 11
    RP_component_phishing = [1, 2, 5, 8, 9]
    clustering_rp(cluster_range,RP_component_phishing,phishing_dataset,assignment3_graphs)

    # RFE + Kmeans and EM
    RFE_component_phishing = [1, 2, 5, 8, 9]
    # for phishing ICA, we can only have 10 Principal components since the number of features for phishing is 10
    clustering_rfp(cluster_range,RFE_component_phishing,phishing_dataset,assignment3_graphs)
