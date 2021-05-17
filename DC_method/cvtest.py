import cvxpy as cp
import random
from DC_method.util import *
import numpy as np


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


class FusedGraphClustering(object):
    def __init__(self, observed_adjacency_matrix, cluster_number,
                 param_l, param_t, param_T, params_for_norm_based_clustering):
        """
        :param observed_adjacency_matrix:
        :param cluster_number:
        :param param_l:
        :param param_t:
        :param param_T:
        :param params_for_norm_based_clustering: a list of all parameters for norm based clustering
        """
        self.fused_graph_adj = None  # to be created
        self.fused_graph_length = None  # to be created
        self.super_node_dict = None  # to be created
        self.high_confidence_nodes_list = None  # to be created
        self.recovered_adjacency_matrix = None  # to be created
        self.param_T = param_T
        self.param_t = param_t
        self.param_l = param_l
        self.cluster_number = cluster_number
        self.observed_adjacency_matrix = observed_adjacency_matrix
        self.params_for_norm_based_clustering = params_for_norm_based_clustering

    def spectral_clustering(self):
        adjacency_matrix = self.recovered_adjacency_matrix
        # adjacency_matrix = self.fused_graph_adj
        degree = np.sum(adjacency_matrix, axis=1)
        d = np.diag((degree + np.mean(degree)) ** (-0.5))  # 得到度矩阵
        l = np.dot(np.dot(d, adjacency_matrix), d)  # laplace matrix
        u, _, _ = np.linalg.svd(l)
        spectral = u[:, list(range(self.cluster_number))]
        model = KMeans(n_clusters=self.cluster_number)
        model_fit = model.fit(spectral)
        cluster_label_list = list(model_fit.labels_)  # labels (cluster information)
        final_result = [[] for _ in range(self.cluster_number)]
        for kv in self.super_node_dict.items():
            final_result[cluster_label_list[kv[0]]].append(kv[1])
        return final_result

    def norm_based_clustering(self):
        # construct two sets
        adjacency_node_set = []
        adjacency_node_num = len(self.fused_graph_adj)
        for i in range(adjacency_node_num):
            for j in range(adjacency_node_num):
                if self.fused_graph_adj[i][j]:
                    adjacency_node_set.append((i, j))
        confidence_node_set = []
        confidence_node_num = len(self.high_confidence_nodes_list)
        for i in range(confidence_node_num):
            for j in range(confidence_node_num):
                confidence_node_set.append((i, j))
        parameter1 = self.params_for_norm_based_clustering[0]
        parameter2 = self.params_for_norm_based_clustering[1]
        parameter3 = self.params_for_norm_based_clustering[2]

        # ---------------- USE CVXPY TO SOLVE --------------------- #
        total_node_num = self.fused_graph_length
        all_node_set = [(i, j) for i in range(total_node_num) for j in range(total_node_num)]  # 全集
        adjacency_node_oppo_set = list(set(all_node_set).intersection(set(adjacency_node_set)))  # A的补
        confidence_node_oppo_set = list(set(all_node_set).intersection(set(confidence_node_set)))  # C的补
        adjacency_intersect_confidence_oppo_set = list(set(adjacency_node_set).intersection(set(confidence_node_oppo_set)))  # A交C的补
        adjacency_oppo_intersect_confidence_oppo_set = list(set(adjacency_node_oppo_set).intersection(set(confidence_node_oppo_set)))  # A的补交C的补
        # 第一个投影矩阵
        proj_adjacency_intersect_confidence_oppo_matrix = np.zeros((total_node_num, total_node_num))
        for item in adjacency_intersect_confidence_oppo_set:
            proj_adjacency_intersect_confidence_oppo_matrix[item[0]][item[1]] = 1
        # 第二个投影矩阵
        proj_adjacency_oppo_intersect_confidence_oppo_matrix = np.zeros((total_node_num, total_node_num))
        for item in adjacency_oppo_intersect_confidence_oppo_set:
            proj_adjacency_oppo_intersect_confidence_oppo_matrix[item[0]][item[1]] = 1
        # 第三个投影矩阵
        proj_confidence_matrix = np.zeros((total_node_num, total_node_num))
        for item in confidence_node_set:
            proj_confidence_matrix[item[0]][item[1]] = 1

        # Construct the problem.
        y_matrix = cp.Variable((total_node_num, total_node_num))
        s_matrix = cp.Variable((total_node_num, total_node_num))

        objective = cp.Minimize(cp.normNuc(y_matrix) +
                                parameter1 * cp.atoms.sum(cp.abs(
            cp.multiply(proj_adjacency_intersect_confidence_oppo_matrix, s_matrix))) +
                                parameter2 * cp.atoms.sum(cp.abs(
            cp.multiply(proj_adjacency_oppo_intersect_confidence_oppo_matrix, s_matrix))) +
                                parameter3 * cp.atoms.sum(cp.abs(
            cp.multiply(proj_confidence_matrix, s_matrix))))
        constraints = [0 <= y_matrix, y_matrix <= 1, s_matrix + y_matrix == self.fused_graph_adj]

        prob = cp.Problem(objective, constraints)
        print("=>>>>> Optimal value", prob.solve(), "<<<<<=")
        self.recovered_adjacency_matrix = y_matrix.value

        # --------------------------------------------------------- #

    def build_fused_graph(self, recovered_cluster_list):
        # Break up small clusters
        clusters_list = recovered_cluster_list
        for cluster in clusters_list:
            clusters_list.remove(cluster)
            if len(cluster) < self.param_T:
                for item in cluster:
                    clusters_list.append([item])
            else:
                split = split_list(cluster, self.param_l)
                for item in split:
                    clusters_list.append(item)

        # Create Super Nodes
        node_num = len(clusters_list)
        super_node_dict = dict()
        for i in range(node_num):
            super_node_dict[i] = clusters_list[i]

        # Build the fused graph
        self.fused_graph_adj = np.zeros((node_num, node_num))
        self.fused_graph_length = node_num

        def findByRow(mat, which_row):
            return np.where((mat == which_row).all(1))[0]

        for i in range(node_num):
            node1 = super_node_dict[i]
            for j in range(node_num):
                node2 = super_node_dict[j]
                if len(node1) + len(node2) == 2:
                    if len(findByRow(self.observed_adjacency_matrix, np.array([node1[0], node2[0]]))):
                        self.fused_graph_adj[i][j] = 1
                else:
                    s = 0
                    for item1 in node1:
                        for item2 in node2:
                            if len(findByRow(self.observed_adjacency_matrix,
                                             np.array([item1, item2]))):
                                s += 1
                        E_hat = s / (len(node1) * len(node2))
                        if E_hat >= self.param_t:
                            self.fused_graph_adj[i][j] = 1
                        else:
                            self.fused_graph_adj[i][j] = 0
        # Construct the set of high confidence nodes
        high_confidence_nodes_list = [x for x in range(node_num) if len(clusters_list[x]) > 1]
        self.super_node_dict = super_node_dict
        self.high_confidence_nodes_list = high_confidence_nodes_list




if __name__ == '__main__':
    # just for Test
    param1 = 1
    param2 = 1
    param3 = 1
    total_node_number = 5
    A_set = [(0, 1), (0, 2), (1, 0), (2, 0), (1, 3), (3, 1), (4, 2), (2, 4)]
    C_set = [(0, 1), (1, 0), (1, 3), (3, 1), (0, 3), (3, 0)]
    adj_mat = np.zeros((5, 5))
    for item in A_set:
        adj_mat[item[0]][item[1]] = 1
    # res = solve_fused_graph(total_node_number, param1, param2, param3, A_set, C_set, adj_mat)
    # print(res)
