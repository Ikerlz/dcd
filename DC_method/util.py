import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import itertools
import findspark
import pyspark
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import *
import time


def simulate_sbm_dc_data(sbm_matrix, sample_size=1000, partition_num=10, cluster_num=3):
    """
    :param sbm_matrix:
    :param sample_size:
    :param partition_num:
    :param cluster_num:
    :return:
    """
    if (sbm_matrix.shape[0] != cluster_num) | \
            (sbm_matrix.shape[1] != cluster_num) | \
            (sbm_matrix.shape[0] != sbm_matrix.shape[1]):
        raise Exception("sbm_matrix shape Error or the Shape is not equal to Cluster_num")
    else:
        data_index = [x for x in range(sample_size)]
        data_cluster = np.random.randint(0, cluster_num, sample_size).tolist()
        index_cluster = dict(zip(data_index, data_cluster))
        X = np.empty(shape=[0, 3], dtype=int)
        X = np.append(X, [[0, -1, np.random.randint(0, partition_num, 1)[0]]], axis=0)
        for i in range(1, sample_size):
            p_num = np.random.randint(0, partition_num, 1)[0]
            X = np.append(X, [[i, -1, p_num]], axis=0)  # to avoid node lost
            for j in range(i):
                if np.random.binomial(1, sbm_matrix[index_cluster[i], index_cluster[j]], 1):
                    X = np.append(X, [[i, j, p_num]], axis=0)
        data_pdf = pd.DataFrame(X, columns=["IndexNum1"] + ["IndexNum2"] + ["PartitionID"])
        return data_pdf, index_cluster


def get_laplace_matrix(adjacency_matrix, position="master", regularization=False):
    """
    :param adjacency_matrix: 邻接矩阵（方阵或长矩阵）
    :param position: master或worker
    :param regularization: 是否进行正则化
    :return: 拉普拉斯矩阵
    """
    if regularization:
        if position == "master":
            degree = np.sum(adjacency_matrix, axis=1)
            d = np.diag((degree + np.mean(degree)) ** (-0.5))  # 得到度矩阵
            return np.dot(np.dot(d, adjacency_matrix), d)

        elif position == "worker":

            # 2020.7.18 for test
            out_degree = np.sum(adjacency_matrix, axis=1)
            out_degree_matrix = np.diag((out_degree + np.mean(out_degree)) ** (-0.5))
            for i in range(out_degree_matrix.shape[0]):
                if out_degree_matrix[i, i] == np.infty:
                    out_degree_matrix[i, i] = 1000
            in_degree = np.sum(adjacency_matrix, axis=0)
            in_degree_matrix = np.diag((in_degree + np.mean(in_degree)) ** (-0.5))
            ###
            laplace_matrix = np.dot(np.dot(out_degree_matrix, adjacency_matrix), in_degree_matrix)

            return laplace_matrix

            # D = np.diag(np.sum(adjacency_matrix, axis=1) ** (-0.5))
            # F = np.diag(np.sum(adjacency_matrix, axis=0) ** (-0.5))
            # return np.dot(np.dot(D, adjacency_matrix), F)  # 得到度矩阵

        else:
            raise Exception("Input Error: worker or master is expected but {} are given".format(position))
    else:
        if position == "master":
            d = np.diag(np.sum(adjacency_matrix, axis=1) ** (-0.5))  # 得到度矩阵
            return np.dot(np.dot(d, adjacency_matrix), d)

        elif position == "worker":
            out_degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1) ** (-0.5))
            for i in range(out_degree_matrix.shape[0]):
                if out_degree_matrix[i, i] == np.infty:
                    out_degree_matrix[i, i] = 10000
            in_degree_matrix = np.diag(np.sum(adjacency_matrix, axis=0) ** (-0.5))
            laplace_matrix = np.dot(np.dot(out_degree_matrix, adjacency_matrix), in_degree_matrix)

            return laplace_matrix

            # D = np.diag(np.sum(adjacency_matrix, axis=1) ** (-0.5))
            # F = np.diag(np.sum(adjacency_matrix, axis=0) ** (-0.5))
            # return np.dot(np.dot(D, adjacency_matrix), F)  # 得到度矩阵

        else:
            raise Exception("Input Error: worker or master is expected but {} are given".format(position))


def get_spectral(laplace_matrix, k, normalization=False, method='svd'):
    """
    :param laplace_matrix: 拉普拉斯矩阵
    :param k: 截取SVD后的前k个向量
    :param normalization: 是否归一化
    :param method: 选择用奇异值分解（SVD）还是特征值分解（EVD）
    :return: 得到的谱
    """
    if method == 'svd':
        u, _, _ = np.linalg.svd(laplace_matrix)
        spectral = u[:, list(range(k))]
        if normalization:
            row_len = len(u)  # 行数
            for i in range(row_len):
                norm2 = np.linalg.norm(spectral[i])
                if norm2:
                    spectral[i] = spectral[i] / np.linalg.norm(spectral[i])
    elif method == 'evd':
        e_vals, e_vecs = np.linalg.eig(laplace_matrix)
        sorted_indices = np.argsort(e_vals)
        spectral = e_vecs[:, sorted_indices[:-k-1:-1]]
        if normalization:
            row_len = len(e_vecs)  # 行数
            for i in range(row_len):
                norm2 = np.linalg.norm(spectral[i])
                if norm2:
                    spectral[i] = spectral[i] / np.linalg.norm(spectral[i])
    else:
        raise ValueError("method must be 'svd' or 'evd' but {} is given".format(method))

    return spectral


def worker_clustering(worker_df, cluster_num):
    """
    :param worker_df:
    :param method:
    :param cluster_num:
    :return:
    """
    node_list = list(set(worker_df["IndexNum1"].tolist()))
    node_num = len(node_list)
    index_list = [x for x in range(node_num)]
    node2index = dict(zip(node_list, index_list))
    adj_matrix = np.zeros((node_num, node_num), dtype=int)
    for i in range(node_num):
        adj_matrix[i][i] = 10
    for row in worker_df.itertuples(index=False, name='Pandas'):
        item1 = getattr(row, "IndexNum1")
        item2 = getattr(row, "IndexNum2")
        if (item2 in node_list) & (item2 != -1):
            adj_matrix[node2index[item1]][node2index[item2]] = 1
            adj_matrix[node2index[item2]][node2index[item1]] = 1

    # first, get the laplace matrix
    laplace_matrix = get_laplace_matrix(adj_matrix,
                                        position='master',
                                        regularization=False)

    # second, get the spectral
    spectral = get_spectral(laplace_matrix, cluster_num, normalization=False, method='svd')

    # third, do k-means in spectral
    model = KMeans(n_clusters=cluster_num)
    model_fit = model.fit(spectral)  # do k_means in spectral_transpose
    # cluster_center = model_fit.cluster_centers_  # center points
    cluster_label = list(model_fit.labels_)  # labels (cluster information)
    # return
    worker_num = worker_df["PartitionID"].tolist()[0]
    out_df = pd.DataFrame({"PartitionID": [worker_num for _ in range(len(node_list))],
                           "IndexNum": node_list,
                           "ClusterExp": cluster_label})
    return out_df


def get_accurate(clustering_res_df, cluster_number, error=False):
    """
    :param clustering_res_df: a pandas DataFrame about clustering result
    :param cluster_number: the number of the cluster
    (the first column is the index,
    the second column is the right information,
    the third column is the clustering information)
    :param error: if error=True, then return the error rate, else, return the accuracy rate
    :return: the clustering accuracy
    """
    if clustering_res_df.shape[1] != 3:
        raise Exception("Shape Error: the input DataFrame's column number is not 3")
    real_dict = {}
    clustering_dict = {}
    for i in range(cluster_number):
        real_df = clustering_res_df.loc[clustering_res_df['ClusterInfo'] == i]
        clustering_df = clustering_res_df.loc[clustering_res_df['ClusterExp'] == i]
        real_dict[i] = real_df['IndexNum'].tolist()
        clustering_dict[i] = clustering_df['IndexNum'].tolist()

    accuracy_matrix = np.zeros((cluster_number, cluster_number))
    for i in range(cluster_number):
        for j in range(cluster_number):
            accuracy_matrix[i][j] = len(set(real_dict[i]).intersection(set(clustering_dict[j])))
    # for test
    # print("The accuracy matrix is: \n", accuracy_matrix)
    case_iterator = itertools.permutations(range(cluster_number), cluster_number)

    accurate = 0

    for item in case_iterator:
        acc = sum([accuracy_matrix[i][item[i]] for i in range(cluster_number)])
        if acc > accurate:
            accurate = acc
    if not error:
        return accurate / clustering_res_df.shape[0]
    else:
        return 1 - accurate / clustering_res_df.shape[0]




# TODO some SBM matrix


sbm_matrix1 = np.array([[0.7, 0.45, 0.45],
                        [0.45, 0.7, 0.45],
                        [0.45, 0.45, 0.7]])
sbm_matrix2 = np.array([[0.8, 0.4, 0.4],
                        [0.4, 0.8, 0.4],
                        [0.4, 0.4, 0.8]])
sbm_matrix3 = np.array([[0.6, 0.45, 0.45],
                        [0.45, 0.6, 0.45],
                        [0.45, 0.45, 0.6]])
sbm_matrix4 = np.array([[0.2, 0.1, 0.1],
                        [0.1, 0.2, 0.1],
                        [0.1, 0.1, 0.2]])



if __name__ == '__main__':
    # Model Settings
    sbm_matrix = sbm_matrix4
    sample_size = 1000
    master_num = 100
    worker_per_sub = 20
    partition_num = 50
    cluster_num = 3
    a, b = simulate_sbm_dc_data(sbm_matrix)
    c = worker_clustering(a, 3)
    real_label = []
    for row in c.itertuples(index=False, name='Pandas'):
        item = getattr(row, "IndexNum")
        real_label.append(b[item])
    c["ClusterInfo"] = real_label
    print(get_accurate(c, 3))
    print(c)
    # print(a)
