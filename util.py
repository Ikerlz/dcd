import numpy as np
import random
import math
from sklearn.cluster import KMeans
import itertools


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
            # out_degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1) ** (-0.5))
            # for i in range(out_degree_matrix.shape[0]):
            #     if out_degree_matrix[i, i] == np.infty:
            #         out_degree_matrix[i, i] = 1000
            # in_degree_matrix = np.diag(np.sum(adjacency_matrix, axis=0) ** (-0.5))

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
