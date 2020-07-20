import numpy as np
import pandas as pd


def split_master_worker(total_adjacency_matrix, index2label_dict, master_num=50, partition_num=10, random_select=False):
    """
    :param total_adjacency_matrix: the whole network matrix
    :param index2label_dict: the dict contained the real cluster label information of all nodes
    :param master_num: the number of master nodes (pilot nodes)
    :param partition_num: the number of worker (M)
    :param random_select: decide how to select the pilot nodes;
           for real data, "random_select=False" is recommended,
           which means we select the highest degree nodes as the pilot nodes
    :return: a pandas data frame contained the worker information and a data frame contained master information
    """
    if total_adjacency_matrix.shape[0] != total_adjacency_matrix.shape[1]:
        raise Exception('The shape of the matrix is not correct.')
    else:
        index_list = list(index2label_dict.keys())

        # get master information
        if random_select:
            master_index = list(np.random.choice(index_list, master_num, replace=False))
        else:
            degree_list = np.sum(total_adjacency_matrix, axis=1).tolist()
            index_degree_dict = dict(zip(index_list, degree_list))
            sort_degree_list = sorted(index_degree_dict.items(), key=lambda item: item[1], reverse=True)
            master_index = [item[0] for item in sort_degree_list[0:master_num]]

        # construct the adjacency matrix of master
        adjacency_matrix_master_rows = total_adjacency_matrix[master_index]
        adjacency_matrix_master = adjacency_matrix_master_rows[:, master_index]
        master_cluster_info = [index2label_dict[x] for x in master_index]
        data_master_np = np.concatenate((np.array(master_index, dtype=int).reshape(master_num, 1),
                                         np.array(master_cluster_info, dtype=int).reshape(master_num, 1),
                                         adjacency_matrix_master), 1)
        data_master_pdf = pd.DataFrame(data_master_np, columns=["IndexNum"] +
                                                               ["ClusterInfo"] +
                                                               [str(x) for x in master_index])

        # get worker information
        # here we need to construct a pandas data frame, the first column is the "PartitionID",
        # which is used for partition in spark; the second column is the "IndexNum";
        # the third line is the "ClusterInfo", which represent the true clustering information;
        # then, other columns is the adjacency_matrix

        worker_total_num = total_adjacency_matrix.shape[0] - master_num
        worker_index = [x for x in index_list if x not in master_index]
        worker_cluster_info = [index2label_dict[x] for x in worker_index]

        adjacency_matrix_worker_rows = total_adjacency_matrix[worker_index]
        adjacency_matrix_worker = adjacency_matrix_worker_rows[:, master_index]

        # adjacency_matrix_worker = np.zeros((worker_total_num, master_num), dtype=int)
        # for i in range(worker_total_num):
        #     for j in range(master_num):
        #         adjacency_matrix_worker[i, j] = adjacency_matrix[worker_index[i], master_index[j]]

        partition_id = np.random.randint(0, partition_num, worker_total_num, dtype=int).reshape(worker_total_num, 1)
        data_worker_np = np.concatenate((partition_id,
                                         np.array(worker_index, dtype=int).reshape(worker_total_num, 1),
                                         np.array(worker_cluster_info, dtype=int).reshape(worker_total_num, 1),
                                         adjacency_matrix_worker), 1)
        data_worker_pdf = pd.DataFrame(data_worker_np, columns=["PartitionID"] +
                                                               ["IndexNum"] +
                                                               ["ClusterInfo"] +
                                                               [str(x) for x in master_index])
        return data_master_pdf, data_worker_pdf





if __name__ == '__main__':
    node_df = pd.read_csv('node_index_label.csv')
    node_relationship = pd.read_csv('node_relationship.csv')
    index_list = list(node_df.iloc[:, 2])
    label_list = list(node_df.iloc[:, 1])
    id_list = list(node_df.iloc[:, 0])

    index2id_dict = dict(zip(index_list, id_list))
    index2label_dict = dict(zip(index_list, label_list))

    # adjacency matrix
    pumbed_adjacency_matrix = np.zeros((19717, 19717), dtype=int)

    for i in range(19717):
        pumbed_adjacency_matrix[i, i] = 10

    for row in node_relationship.itertuples():
        index1 = getattr(row, 'index1')
        index2 = getattr(row, 'index2')
        if index1 != index2:
            pumbed_adjacency_matrix[index1, index2] = 1
            pumbed_adjacency_matrix[index2, index1] = 1
    a = split_master_worker(pumbed_adjacency_matrix, index2label_dict, master_num=500)
    b = split_master_worker(pumbed_adjacency_matrix, index2label_dict, master_num=500, random_select=True)
    print(a)