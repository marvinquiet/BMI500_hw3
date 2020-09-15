import os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

class My_KMeans:
    '''Implement KMeans on my own
    '''
    def __init__(self, K = 7, max_iter=1000, tol=1e-04, random_state=2020,
            method="Euclidean"):
        self.K = K
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.method = method

    def distance(self, a, b):
        if self.method == "Euclidean":
            return np.linalg.norm(a-b, axis=0)
        elif self.method == "Manhattan":
            return np.sum(np.abs(a-b), axis=0)

    def fit(self, X):
        X = pd.DataFrame(X) # turn into pandas dataframe in case numpy array

        nrow = X.shape[0]
        # np.random.seed(self.random_state)
        # self.centroids = X.iloc[np.random.randint(nrow, size=self.K), ]

        # initialize centroids (random choose)
        self.centroids = X.sample(self.K, random_state=self.random_state)
        # self.centroids = X.iloc[np.random.randint(nrow, size=self.K), ]
        self.centroids.reset_index(drop=True, inplace=True) # remove index

        iter_n = 0
        tol_n = 1e5

        self.tols = []
        while iter_n < self.max_iter and tol_n > self.tol: # threshold for stopping criteria
            # initialize labels
            self.labels = [0] * nrow

            for i in range(nrow):
                data_i = X.iloc[i, ]

                min_dist = np.Inf
                for k in range(self.centroids.shape[0]):
                    centroid = self.centroids.iloc[k, ]
                    dist_centroid = self.distance(centroid, data_i)

                    if dist_centroid < min_dist: # if the minimum, change the labels
                        min_dist = dist_centroid
                        self.labels[i] = k

            tmp_centroids = self.centroids.copy() # mark for calculating the difference
            for label in range(self.K):
                rows = [i for i, value in enumerate(self.labels) if self.labels[i] == label]
                avg_centroid = np.average(X.iloc[rows, ], axis=0) # get the new centroid
                self.centroids.loc[label] = avg_centroid

            tol_n = sum(self.distance(tmp_centroids, self.centroids)) # calculate differencen between two centroids
            # for plotting
            self.tols.append(tol_n)
            # print("iter:", iter_n, "diff:", tol_n)

            iter_n += 1

def gene_selection(data):
    count_nonzeros = (data!=0).sum(axis=1)
    genes = count_nonzeros[count_nonzeros > 1200].index.tolist()
    filtere_data = data.loc[genes, ]
    return filtere_data

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
    reducer = umap.UMAP(random_state=2020)
    embedding = reducer.fit_transform(data.T)

    # original cell-type annotation
    # plt.scatter(embedding[:, 0], embedding[:, 1], c=[celltype_color[l] for l in celltype_list], s=0.5)
    # plt.savefig("golden_standard.png", )

    # sub_celltypes = pd.read_csv("FC_celltypes.csv", header=0, index_col=0) # sub-cell type file

    from sklearn.cluster import KMeans
    # try sklearn's Kmeans on pre-defined 9 clusters

    # start_time = time.time()
    # kmeans = KMeans(n_clusters=9, random_state=2020).fit(data.T)
    # print("sklearn KMeans: ", time.time()-start_time)

    # plt.scatter(embedding[:, 0], embedding[:, 1], c=[list(celltype_color.values())[kl] for kl in kmeans.labels_], s=0.5)
    # plt.savefig("scikitlearn_kmeans_ori_data.png")

    # try my KMeans
    # start_time = time.time()
    # my_kmeans = My_KMeans(K=9, random_state=2020)
    # my_kmeans.fit(data.T)
    # print("My KMeans: ", time.time()-start_time)

    # plt.scatter(embedding[:, 0], embedding[:, 1], c=[list(celltype_color.values())[kl] for kl in my_kmeans.labels], s=0.5)
    # plt.savefig("my_kmeans_ori_data.png")

    # start_time = time.time()
    # my_kmeans = My_KMeans(K=9, random_state=2020, method="Manhattan")
    # my_kmeans.fit(data.T)
    # print("My KMeans mahattan: ", time.time()-start_time)
    # plt.scatter(embedding[:, 0], embedding[:, 1], c=[list(celltype_color.values())[kl] for kl in my_kmeans.labels], s=0.5)
    # plt.savefig("my_kmeans_ori_data_manhattan.png")

    # my_kmeans_embed = My_KMeans(K=9, random_state=2020)
    # my_kmeans_embed.fit(embedding)
    # plt.scatter(embedding[:, 0], embedding[:, 1], c=[list(celltype_color.values())[kl] for kl in my_kmeans_embed.labels], s=0.5)
    # plt.savefig("my_kmeans_embedding.png")

    # gene selection
    filtered_data = gene_selection(data)
    # try my KMeans
    start_time = time.time()
    my_kmeans = My_KMeans(K=9, random_state=2020)
    my_kmeans.fit(filtered_data.T)
    print("My KMeans: ", time.time()-start_time)

    plt.scatter(embedding[:, 0], embedding[:, 1], c=[list(celltype_color.values())[kl] for kl in my_kmeans.labels], s=0.5)
    plt.savefig("my_kmeans_30per_data.png")

    start_time = time.time()
    my_kmeans = My_KMeans(K=9, random_state=2020, method="Manhattan")
    my_kmeans.fit(filtered_data.T)
    print("My KMeans mahattan: ", time.time()-start_time)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=[list(celltype_color.values())[kl] for kl in my_kmeans.labels], s=0.5)
    plt.savefig("my_kmeans_30per_manhattan.png")

    # PCA projection
    from sklearn.decomposition import PCA
    pca = PCA(n_components=15)
    pca.fit(data)
    my_kmeans = My_KMeans(K=9, random_state=2020, method="Euclidean")
    my_kmeans.fit(pca.components_.T)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=[list(celltype_color.values())[kl] for kl in my_kmeans.labels], s=0.5)
    plt.savefig("my_kmeans_pca.png")
