# load packages
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from util import *
import itertools
import findspark
import pyspark
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import *
import time
# simulate data based on SBM model
# TODO write a function to simulate SBM data


def simulate_sbm_data(sbm_matrix, sample_size=1000, master_num=50, partition_num=10, cluster_num=3):
    """
    :param sbm_matrix: the matrix of SBM model
    :param sample_size: the sample size
    :param master_num: the points number in master
    :param partition_num: the partition number, which equal to worker number
    :param cluster_num: the clustering number
    :return: a pandas data frame contained the worker information and a data frame contained master information
    """
    if (sbm_matrix.shape[0] != cluster_num) | \
            (sbm_matrix.shape[1] != cluster_num) | \
            (sbm_matrix.shape[0] != sbm_matrix.shape[1]):
        raise Exception("sbm_matrix shape Error or the Shape is not equal to Cluster_num")
    else:
        data_index = [x for x in range(sample_size)]
        data_cluster = np.random.randint(0, cluster_num, sample_size).tolist()
        index_cluster = dict(zip(data_index, data_cluster))  # {0:1, 1:0, 2:4, ...}

        # construct the whole adjacency matrix
        adjacency_matrix = np.zeros((sample_size, sample_size))
        adjacency_matrix[0, 0] = 10
        for i in range(1, sample_size):
            adjacency_matrix[i, i] = 10
            for j in range(i):
                adjacency_matrix[i, j] = np.random.binomial(1, sbm_matrix[index_cluster[i], index_cluster[j]], 1)
                adjacency_matrix[j, i] = adjacency_matrix[i, j]


        # get master information
        master_index = list(np.random.choice(data_index, master_num, replace=False))  # random select
        adjacency_matrix_master_rows = adjacency_matrix[master_index]
        adjacency_matrix_master = adjacency_matrix_master_rows[:, master_index]
        # adjacency_matrix_master = np.zeros((master_num, master_num), dtype=int)
        # for i in range(master_num):
        #     for j in range(master_num):
        #         adjacency_matrix_master[i, j] = adjacency_matrix[master_index[i], master_index[j]]

        # here we construct a pandas data frame to store master information,
        # the first column is the "IndexNum";
        # the second column is the "ClusterInfo";
        # other columns represent the adjacency_matrix_master
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

        # TODO return data_worker_pdf [(sample_size - master_num) by (3 + master_num)]

        return data_master_pdf, data_worker_pdf

# TODO test simulate_sbm_data function: succeed


# m_pdf, w_pdf = simulate_sbm_data(np.array([[0.8,0.3,0.3],[0.3,0.8,0.3],[0.3,0.3,0.8]]), cluster_num=3)


########################################################################################################################
########################################################################################################################

# TODO spectral clustering in master

def spectral_clustering_master(master_pdf, cluster_num=3, real_data=False):
    """
    :param master_pdf: a pandas data frame contained the information about the master
    :param cluster_num: the clustering number
    :param real_data: check if the data is real data
    :return: a pseudo dict, a pandas DataFrame about clustering result and the clustering time
    """
    master_index = master_pdf["IndexNum"].tolist()
    # TODO decide whether to calculate the accurate rate or not
    # master_cluster_info = master_pdf["ClusterInfo"].tolist()
    adjacency_matrix_master = master_pdf.iloc[:, 2:]

    # time start
    start_time = time.time()

    if real_data:
        # first, get the laplace matrix
        laplace_matrix = get_laplace_matrix(adjacency_matrix_master,
                                            position='master',
                                            regularization=True)
        # second, get the spectral
        spectral = get_spectral(laplace_matrix, cluster_num, normalization=True, method='svd')
    else:
        # first, get the laplace matrix
        laplace_matrix = get_laplace_matrix(adjacency_matrix_master,
                                            position='master',
                                            regularization=False)

        # second, get the spectral
        spectral = get_spectral(laplace_matrix, cluster_num, normalization=False, method='svd')
        # u, sigma, v_transpose = np.linalg.svd(laplace_matrix)
        # spectral = u[:, list(range(cluster_num))]

    # third, do k-means in spectral
    model = KMeans(n_clusters=cluster_num)
    model_fit = model.fit(spectral)  # do k_means in spectral_transpose
    cluster_center = model_fit.cluster_centers_  # center points
    cluster_label = list(model_fit.labels_)  # labels (cluster information)

    # forth, calculate the distance to center points
    row = spectral.shape[0]  # row = len(master_index)
    column = spectral.shape[1]  # column = cluster_num
    distance_matrix = np.zeros((row, column))

    for i in range(row):
        for j in range(column):
            # calculate the Euclidean distance between the mapping points and the cluster centers
            distance_matrix[i, j] = np.sqrt(
                sum(np.power(spectral[i] - cluster_center[j], 2)))

    # finally, find the pseudo center and return
    pseudo_center_dict = {}
    for i in range(cluster_num):
        # find the minimum distance to the center point
        index = list(np.argwhere(distance_matrix[:, i] == min(distance_matrix[:, i])))
        index = int(index[0])
        pseudo_center_dict[master_index[index]] = cluster_label[index]

    # time end
    end_time = time.time()

    # return the result of the clustering on the master (a pandas DataFrame)

    out_df = pd.DataFrame(master_pdf, columns=["IndexNum", "ClusterInfo"])
    out_df["ClusterExp"] = cluster_label

    running_time = end_time - start_time

    # TODO decide whether to get the error number or not
    return pseudo_center_dict, out_df, running_time

# TODO test "spectral_clustering_master" function: succeed
# print(spectral_clustering_master(m_pdf))
# {316.0: 0, 111.0: 1, 309.0: 2}

########################################################################################################################
########################################################################################################################

# TODO spectral clustering in workers


def clustering_worker(worker_pdf, master_pdf, pseudo_center_dict, real_data=False):
    """
    :param worker_pdf: the pandas data frame contained worker information
    :param master_pdf: the pandas data frame contained master information
    :param pseudo_center_dict: the result of clustering on master like {316.0: 0, 111.0: 1, 309.0: 2}
    :param real_data: check if the data is real data
    :return: a pandas data frame (three columns, first column is the index,
    second column is the true cluster labels, third column is the cluster labels in experiment)
    """
    master_index = master_pdf["IndexNum"].tolist()
    worker_index = worker_pdf["IndexNum"].tolist()
    total_index = master_index + worker_index
    pseudo_index = list(pseudo_center_dict.keys())
    adjacency_matrix_master = master_pdf.iloc[:, 2:]
    adjacency_matrix_worker = worker_pdf.iloc[:, 3:]
    adjacency_matrix = np.vstack((adjacency_matrix_master, adjacency_matrix_worker))
    # [len(master_index)+len(worker_index)]-by-[len(pseudo_center_dict.keys())]

    # time start
    # start_time = time.time()
    if real_data:
        t1 = time.time()
        # first, get the laplace matrix
        laplace_matrix = get_laplace_matrix(adjacency_matrix,
                                            position='worker',
                                            regularization=True)
        # second, get the spectral
        spectral = get_spectral(laplace_matrix, len(pseudo_index), normalization=True, method='svd')
    else:
        t1 = time.time()
        # first, get the laplace matrix
        laplace_matrix = get_laplace_matrix(adjacency_matrix,
                                            position='worker',
                                            regularization=False)
        # second, get the spectral
        spectral = get_spectral(laplace_matrix, len(pseudo_index), normalization=False, method='svd')

    # first, get the laplace matrix
    # out_degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1) ** (-0.5))
    # in_degree_matrix = np.diag(np.sum(adjacency_matrix, axis=0) ** (-0.5))
    # laplace_matrix = np.dot(np.dot(out_degree_matrix, adjacency_matrix), in_degree_matrix)

    # second, get the spectral
    # u, sigma, v_transpose = np.linalg.svd(laplace_matrix)
    # spectral = u[:, list(range(len(pseudo_index)))]

    # third, clustering on the spectral
    worker_cluster_list = []
    pseudo_index_in_total_index = [total_index.index(key) for key in pseudo_index]
    for i in range(len(total_index)):
        # to store the distance between point mapping and each pseudo_center
        distance_list = [np.sqrt(sum(np.power(
            spectral[index] - spectral[i], 2)))
            for index in pseudo_index_in_total_index]
        if total_index[i] not in master_index:
            worker_cluster_list.append(pseudo_center_dict[pseudo_index[distance_list.index(min(distance_list))]])
    t2 = time.time()
    print(round(t2-t1, 6))

    # time end
    # end_time = time.time()

    # finally, return the pandas data frame
    out_df = pd.DataFrame(worker_pdf, columns=["IndexNum", "ClusterInfo"])
    out_df["ClusterExp"] = worker_cluster_list
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    print(out_df)

    # running_time = end_time - start_time

    return out_df

# TODO test "clustering_worker" function: succeed


# m_pdf, w_pdf = simulate_sbm_data(np.array([[0.8, 0.3, 0.3],
#                                           [0.3, 0.8, 0.3],
#                                            [0.3, 0.3, 0.8]]), master_num=200, cluster_num=3)

# pcd = spectral_clustering_master(m_pdf)
# a = clustering_worker(w_pdf, m_pdf, pcd)
# print(a)

# TODO

# TODO Calculate clustering accuracy


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


# for real data
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
########################################################################################################################
########################################################################################################################

# Main

# if __name__ == '__main__':
#     # Spark Environment
#     findspark.init("/usr/local/spark")
#     spark = pyspark.sql.SparkSession.builder.appName("Spark Spectral Clustering").getOrCreate()
#
#     # Pandas options
#     pd.options.display.max_columns = None
#     pd.options.display.max_rows = None
#     np.set_printoptions(threshold=np.inf)
#
#     # Model Settings
#     sbm_matrix = np.array([[0.8, 0.3, 0.3],
#                            [0.3, 0.8, 0.3],
#                            [0.3, 0.3, 0.8]])
#     sample_size = 2000
#     master_num = 500
#     partition_num = 50
#     cluster_num = 3
#
#     # Main
#     start0 = time.time()
#     print("simulate data")
#     master_pdf, worker_pdf = simulate_sbm_data(sbm_matrix, sample_size, master_num, partition_num, cluster_num)
#     print("start")
#     start = time.time()
#     master_pseudo_dict = spectral_clustering_master(master_pdf, cluster_num)
#     worker_sdf = spark.createDataFrame(worker_pdf)  # Convert Pandas DataFrame to Spark DataFrame
#     worker_sdf = worker_sdf.repartitionByRange("PartitionID")
#     # worker_sdf = worker_sdf.repartition(partition_num, "PartitionID")
#
#     # Register a user defined function via the Pandas UDF
#     beta = StructType([StructField('Index', FloatType(), True),
#                        StructField('ClusterInfo', FloatType(), True),
#                        StructField('ClusterExp', IntegerType(), True)])
#
#     @pandas_udf(beta, PandasUDFType.GROUPED_MAP)
#     def clustering_worker_udf(data_frame):
#         return clustering_worker(worker_pdf=data_frame,
#                                  master_pdf=master_pdf,
#                                  pseudo_center_dict=master_pseudo_dict)
#     print("woker")
#     start1 = time.time()
#     worker_cluster_sdf = worker_sdf.groupby("PartitionID").apply(clustering_worker_udf)
#     end = time.time()
#     print(end - start)
#     print(end - start1)
#     print(start - start0)
#     a = worker_cluster_sdf.toPandas()
#
#     print(a)
#     # worker_cluster_sdf.show()
