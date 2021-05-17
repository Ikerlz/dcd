#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：dcd 
@File    ：utils.py
@Author  ：Iker Zhe
@Date    ：2021/3/5 19:43 
'''
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import random
import time

def simulate_sbm_dc_data(sbm_matrix, dcd_master_node=None, sample_size=1000, partition_num=10, cluster_num=3, method="norm_based"):
    """
    :param sbm_matrix: the SBM matrix
    :param dcd_master_node: the number of master node if we use dcd method
    :param sample_size: the sample size
    :param partition_num: the worker number
    :param cluster_num: the cluster number
    :param method: 'norm_based', 'dcd' or 'both'
    :return: the pandas pdf
    """
    if (sbm_matrix.shape[0] != cluster_num) | \
            (sbm_matrix.shape[1] != cluster_num) | \
            (sbm_matrix.shape[0] != sbm_matrix.shape[1]):
        raise Exception("sbm_matrix shape Error or the Shape is not equal to Cluster_num")
    else:
        data_index = [x for x in range(sample_size)]
        data_cluster = np.random.randint(0, cluster_num, sample_size).tolist()
        data_partition = np.random.randint(0, partition_num, sample_size).tolist()
        index_cluster = dict(zip(data_index, data_cluster))
        index_partition = dict(zip(data_index, data_partition))
        node_mat = np.array([data_index,
                             [-1 for _ in range(sample_size)],
                             data_partition]).T
        sparse_mat = np.zeros((int(0.5 * sample_size * (sample_size - 1)), 3), dtype=int)
        # construct the whole adjacency matrix
        adjacency_matrix = np.zeros((sample_size, sample_size), dtype=int)
        adjacency_matrix[0, 0] = 1
        row = 0
        for i in range(1, sample_size):
            adjacency_matrix[i, i] = 1
            for j in range(i):
                value = np.random.binomial(1, sbm_matrix[index_cluster[i], index_cluster[j]], 1)
                if value:
                    sparse_mat[row] = [i, j, index_partition[i]]
                    adjacency_matrix[i, j] = adjacency_matrix[j, i] = 1
                    row += 1
        if method == "norm_based":
            info_mat = np.vstack((node_mat, sparse_mat))
            norm_based_clustering_mat = info_mat[~np.equal(info_mat[:, 2], 0)]
            data_pdf = pd.DataFrame(norm_based_clustering_mat, columns=["IndexNum1"] + ["IndexNum2"] + ["PartitionID"])
            return data_pdf, index_cluster
        else:
            master_num = dcd_master_node
            # get master information
            master_index = list(np.random.choice(data_index, master_num, replace=False))  # random select
            adjacency_matrix_master_rows = adjacency_matrix[master_index]
            adjacency_matrix_master = adjacency_matrix_master_rows[:, master_index]
            master_cluster_info = [index_cluster[x] for x in master_index]
            data_master_np = np.concatenate((np.array(master_index, dtype=int).reshape(master_num, 1),
                                             np.array(master_cluster_info, dtype=int).reshape(master_num, 1),
                                             adjacency_matrix_master), 1)
            data_master_pdf = pd.DataFrame(data_master_np, columns=["IndexNum"] +
                                                                   ["ClusterInfo"] +
                                                                   [str(x) for x in master_index])

            # TODO return data_master_pdf [(master_num)-by-(2 + master_num)]

            # get worker information
            # here we need to construct a pandas data frame, the first column is the "PartitionID",
            # which is used for partition in spark; the second column is the "IndexNum";
            # the third line is the "ClusterInfo", which represent the true clustering information;
            # then, other columns is the adjacency_matrix

            worker_total_num = sample_size - master_num
            worker_index = [x for x in data_index if x not in master_index]
            worker_cluster_info = [index_cluster[x] for x in worker_index]

            adjacency_matrix_worker_rows = adjacency_matrix[worker_index]
            adjacency_matrix_worker = adjacency_matrix_worker_rows[:, master_index]

            partition_id = np.random.randint(0, partition_num, worker_total_num, dtype=int).reshape(worker_total_num, 1)
            data_worker_np = np.concatenate((partition_id,
                                             np.array(worker_index, dtype=int).reshape(worker_total_num, 1),
                                             np.array(worker_cluster_info, dtype=int).reshape(worker_total_num, 1),
                                             adjacency_matrix_worker), 1)
            data_worker_pdf = pd.DataFrame(data_worker_np, columns=["PartitionID"] +
                                                                   ["IndexNum"] +
                                                                   ["ClusterInfo"] +
                                                                   [str(x) for x in master_index])

            # TODO return data_worker_pdf [(sample_size - master_num) by (3 + master_num)]
            if method == "dcd":
                return data_master_pdf, data_worker_pdf
            elif method == "both":
                info_mat = np.vstack((node_mat, sparse_mat))
                norm_based_clustering_mat = info_mat[~np.equal(info_mat[:, 2], 0)]
                data_pdf = pd.DataFrame(norm_based_clustering_mat,
                                        columns=["IndexNum1"] + ["IndexNum2"] + ["PartitionID"])
                return data_pdf, index_cluster, data_master_pdf, data_worker_pdf
            else:
                raise ValueError("The method should be norm_based, dcd or both, but {} is given".format(method))


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


def norm_based_worker_clustering(worker_df, cluster_num):
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
    start_time = time.time()
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
    end_time = time.time()
    run_time = end_time - start_time
    # for test
    print(run_time)
    # return
    worker_num = worker_df["PartitionID"].tolist()[0]
    out_df = pd.DataFrame({"PartitionID": [worker_num for _ in range(len(node_list))],
                           "IndexNum": node_list,
                           "ClusterExp": cluster_label,
                           "Time": [run_time for _ in range(len(node_list))]})
    return out_df


def split_list(alist, group_num=4, shuffle=True, retain_left=False):
    '''
        用法：将alist切分成group个子列表，每个子列表里面有len(alist)//group个元素
        shuffle: 表示是否要随机切分列表，默认为True
        retain_left: 若将列表alist分成group_num个子列表后还要剩余，是否将剩余的元素单独作为一组
    '''

    index = list(range(len(alist)))  # 保留下标

    # 是否打乱列表
    if shuffle:
        random.shuffle(index)

    elem_num = len(alist) // group_num  # 每一个子列表所含有的元素数量
    res = []

    def subset(alist, idxs):
        '''
            用法：根据下标idxs取出列表alist的子集
            alist: list
            idxs: list
        '''
        sub_list = []
        for idx in idxs:
            sub_list.append(alist[idx])

        return sub_list
    for idx in range(group_num):
        start, end = idx * elem_num, (idx + 1) * elem_num
        res.append(subset(alist, index[start:end]))
    # 是否将最后剩余的元素作为单独的一组
    if group_num * elem_num != len(index):
        remain = subset(alist, index[end:])
        if retain_left:
            res.append(remain)
        else:
            index = np.random.randint(0, len(res))
            res[index] = res[index] + remain
    return res


