import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.cluster import KMeans


class My_KMeans:
    '''Implement KMeans on my own
    '''
    def __init__(self, K = 7, max_iter=1000, tol=1e-05):
        self.K = K
        self.max_iter = max_iter
        self.tol = tol

    def distance(a, b, method = "Euclidean"):
        if method == "Euclidean":
            return np.linalg.norm(a-b)

    def fit(X):
        # initialize centroids





if __name__ == "__main__":
    # load in the data
    data = pd.read_csv("sampled_FC_data.csv", header=0, index_col=0)  # count matrix
    annot = pd.read_csv("sampled_FC_annot.csv", header=0, index_col=0) # annotation file
    celltypes = {"FC_1-": "Interneuron", "FC_2-": "Interneuron",
            "FC_3-": "Neuron", "FC_4-": "Neuron", "FC_5-": "Neuron", "FC_6-": "Neuron", "FC_7-": "Neuron",
            "FC_8-": "Astrocyte", "FC_9-": "Oligodendrocyte", "FC_10-": "Polydendrocyte", "FC_11-": "Microglia",
            "FC_12-": "Endothelial", "FC_13-": "Mural", "FC_14-": "Fibroblast"}
    celltype_color = {"Interneuron": 'red', "Neuron": 'green', 
            "Astrocyte": 'black', "Oligodendrocyte": "blue", "Polydendrocyte": "purple", "Microglia": "pink",
            "Endothelial": "yellow", "Mural": "orange", "Fibroblast": "darkcyan"}
    # convert annotation to major cell types
    annot_list = list(annot.loc[:, "sampled_annot"])
    celltype_list = []
    for item in annot_list:
        for cluster_id, cell_type in celltypes.items():
            if cluster_id in item:
                celltype_list.append(cell_type)
                break

    # for visualizing
    import umap
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(data.T)

    # original cell-type annotation
    ori_scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=[celltype_color[l] for l in celltype_list], s=1)
    plt.legend(handles=ori_scatter.legend_elements()[0], labels=celltype_color.keys())

    sub_celltypes = pd.read_csv("FC_celltypes.csv", header=0, index_col=0) # sub-cell type file

    # try sklearn's Kmeans on pre-defined 9 clusters
    kmeans = KMeans(n_clusters=9, random_state=2020).fit(data.T)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=[list(celltype_color.values())[kl] for kl in kmeans.labels_], s=1)




