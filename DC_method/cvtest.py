import cvxpy as cp
import numpy as np
import random

# n = 100
# A = np.random.randn(n, n)


#
# print("Optimal value", prob.solve())
# print("Optimal var")
# print(Y.value)  # A numpy ndarray# .
# print(S.value)


def solve_fused_graph(total_node_num, param_A, param_Ac, param_C, A_set, C_set, adj_matrix):
    """
    :param total_node_num: 所有node的个数
    :param param_A: 参数
    :param param_Ac: 参数
    :param param_C: 参数
    :param A_set: 一个列表
    :param C_set: 一个列表
    :param adj_matrix: 邻接矩阵
    :return: 一个矩阵
    """
    T_set = [(i, j) for i in range(total_node_num) for j in range(total_node_num)] # 全集
    Ac_set = list(set(T_set).intersection(set(A_set)))  # A的补
    Cc_set = list(set(T_set).intersection(set(C_set)))  # C的补
    A_intersect_Cc_set = list(set(A_set).intersection(set(Cc_set))) # A交C的补
    # 第一个投影矩阵
    proj_A_intersect_Cc_matrix = np.zeros((total_node_num, total_node_num))
    for item in A_intersect_Cc_set:
        proj_A_intersect_Cc_matrix[item[0]][item[1]] = 1
    # 第二个投影矩阵
    Ac_intersect_Cc_set = list(set(Ac_set).intersection(set(Cc_set)))  # A的补交C的补
    proj_Ac_intersect_Cc_matrix = np.zeros((total_node_num, total_node_num))
    for item in Ac_intersect_Cc_set:
        proj_Ac_intersect_Cc_matrix[item[0]][item[1]] = 1
    # 第三个投影矩阵
    proj_C_matrix = np.zeros((total_node_num, total_node_num))
    for item in C_set:
        proj_C_matrix[item[0]][item[1]] = 1

    # 构造优化问题
    # Construct the problem.
    Y = cp.Variable((total_node_num, total_node_num))
    S = cp.Variable((total_node_num, total_node_num))

    objective = cp.Minimize(cp.normNuc(Y) +
                            param_A * cp.atoms.sum(cp.abs(cp.multiply(proj_A_intersect_Cc_matrix, S))) +
                            param_Ac * cp.atoms.sum(cp.abs(cp.multiply(proj_Ac_intersect_Cc_matrix, S))) +
                            param_C * cp.atoms.sum(cp.abs(cp.multiply(proj_C_matrix, S))))
    constraints = [0 <= Y, Y <= 1, S + Y == adj_matrix]

    prob = cp.Problem(objective, constraints)
    print("Optimal value", prob.solve())
    return Y.value


def build_fused_graph(observed_adj_mat, clusters_list, l: int, t, T):
    """
    :param observed_adj_mat: 稀疏矩阵的存储方式, N行2列
    :param clusters_list: A list: [[1,2,3], [4,5,6], [7,8,9], ...]
    :param l:
    :param t:
    :param T:
    :return:
    """
    # Break up small clusters
    for cluster in clusters_list:
        clusters_list = clusters_list.remove(cluster)
        if len(cluster) < T:
            for item in cluster:
                clusters_list.append([item])
        else:
            cut_index_list = random.sample(range(len(cluster)), l)
            for i in range(l):
                if not i:
                    clusters_list.append(cluster[:cut_index_list[i]])
                clusters_list.append(cluster[cut_index_list[i-1]:cut_index_list[i]])

    # Create Super Nodes
    node_num = len(clusters_list)
    super_node_dict = dict()
    for i in range(node_num):
        super_node_dict[i] = clusters_list[i]

    # Build the fused graph
    fused_graph_adj = np.zeros((node_num, node_num))

    def findByRow(mat, row):
        return np.where((mat == row).all(1))[0]
    for i in range(node_num):
        node1 = super_node_dict[i]
        for j in range(node_num):
            node2 = super_node_dict[j]
            if len(node1) + len(node2) == 2:
                if len(findByRow(observed_adj_mat, np.array([node1[0], node2[0]]))):
                    fused_graph_adj[i][j] = 1
            else:
                s = 0
                for item1 in node1:
                    for item2 in node2:
                        if len(findByRow(observed_adj_mat, np.array([item1, item2]))):
                            s += 1
                E_hat = s / (len(node1) * len(node2))
                if E_hat >= t:
                    fused_graph_adj[i][j] = 1
                else:
                    fused_graph_adj[i][j] = 0
    # Construct the set of high confidence nodes
    high_confidence_nodes_list = [x for x in range(node_num) if clusters_list[x] > 1]
    return fused_graph_adj, super_node_dict, high_confidence_nodes_list


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
    res = solve_fused_graph(total_node_number, param1, param2, param3, A_set, C_set, adj_mat)
    print(res)