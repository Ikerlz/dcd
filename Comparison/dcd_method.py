#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：dcd 
@File    ：dcd_method.py
@Author  ：Iker Zhe
@Date    ：2021/3/6 14:09 
'''

from pyspark.sql.functions import pandas_udf, PandasUDFType
from math import ceil
import findspark
import pyspark
from pyspark import SparkConf
from pyspark.sql.types import *
from utils import *
import time
import os
import itertools


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

    # time start
    start_time = time.time()
    if real_data:
        # first, get the laplace matrix
        laplace_matrix = get_laplace_matrix(adjacency_matrix,
                                            position='worker',
                                            regularization=True)
        # second, get the spectral
        spectral = get_spectral(laplace_matrix, len(pseudo_index), normalization=True, method='svd')
    else:
        # first, get the laplace matrix
        laplace_matrix = get_laplace_matrix(adjacency_matrix,
                                            position='worker',
                                            regularization=False)
        # second, get the spectral
        spectral = get_spectral(laplace_matrix, len(pseudo_index), normalization=False, method='svd')

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
    end_time = time.time()
    run_time = end_time - start_time
    # for test
    print(run_time)

    # finally, return the pandas data frame

    out_df = pd.DataFrame(worker_pdf, columns=["IndexNum", "ClusterInfo"])
    out_df["ClusterExp"] = worker_cluster_list
    # insert running time
    out_df['Time'] = [run_time for _ in range(len(worker_cluster_list))]
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


def dcd_clustering(args_dict, data_dict):
    # Model Settings
    sample_size = args_dict["sample_size"]
    cluster_num = args_dict["cluster_num"]
    master_num = args_dict["master_num"]
    worker_per_sub = args_dict["worker_per_sub"]
    master_pdf = data_dict["master_pdf"]
    worker_pdf = data_dict["worker_pdf"]
    #################################################

    # Main
    # Register a user defined function via the Pandas UDF
    beta = StructType([StructField('IndexNum', IntegerType(), True),
                       StructField('ClusterInfo', IntegerType(), True),
                       StructField('ClusterExp', IntegerType(), True),
                       StructField('Time', FloatType(), True)])

    @pandas_udf(beta, PandasUDFType.GROUPED_MAP)
    def clustering_worker_udf(data_frame):
        return clustering_worker(worker_pdf=data_frame,
                                 master_pdf=master_pdf,
                                 pseudo_center_dict=master_pseudo_dict)

    # print(">>>>>>>>>>>>>>>  Master聚类开始  <<<<<<<<<<<<<<<")
    master_pseudo_dict, master_cluster_pdf, master_time = spectral_clustering_master(master_pdf, cluster_num)
    # print(">>>>>>>>>>>>>>>  Master聚类结束，用时{}s  <<<<<<<<<<<<<<<".format(master_time))
    worker_size = sample_size - master_num
    sub_num = ceil(worker_size / worker_per_sub)
    os.environ['JAVA_HOME'] = '/usr/lib/jvm/jdk8u252'
    # Spark Environment
    findspark.init("/usr/local/spark")
    conf = SparkConf(). \
        setMaster("local[*]"). \
        setAll([('spark.executor.memory', '6g'),
                ('spark.driver.memory', '40g')])
    spark = pyspark.sql.SparkSession.builder. \
        config(conf=conf). \
        appName("SparkSC"). \
        getOrCreate()
    for i in range(sub_num):
        if i != sub_num - 1:
            worker_sdf_isub = spark.createDataFrame(worker_pdf[(worker_per_sub * i): (worker_per_sub * (i + 1))])
        else:
            worker_sdf_isub = spark.createDataFrame(worker_pdf[(worker_per_sub * i): worker_size])
        if i == 0:
            worker_sdf = worker_sdf_isub
        else:
            worker_sdf = worker_sdf.unionAll(worker_sdf_isub)
    # worker_sdf = spark.createDataFrame(worker_pdf)  # Convert Pandas DataFrame to Spark DataFrame
    worker_sdf = worker_sdf.repartitionByRange("PartitionID")
    # worker_sdf = worker_sdf.repartition(partition_num, "PartitionID")
    # print("worker")
    # start1 = time.time()
    # print(">>>>>>>>>>>>>>>  Worker聚类开始  <<<<<<<<<<<<<<<")
    worker_cluster_sdf = worker_sdf.groupby("PartitionID").apply(clustering_worker_udf)
    # print(">>>>>>>>>>>>>>>  Spark DataFrame => Pandas DataFrame  <<<<<<<<<<<<<<<")
    worker_cluster_pdf = worker_cluster_sdf.toPandas()  # Spark DataFrame => Pandas DataFrame
    mean_worker_time = np.mean(list(set(worker_cluster_pdf["Time"].values.tolist())))
    worker_cluster_pdf = worker_cluster_pdf.drop(columns=["Time"])
    # print(">>>>>>>>>>>>>>>  Worker聚类结束，用时{}s  <<<<<<<<<<<<<<<".format(mean_worker_time))
    cluster_pdf = pd.concat(
        [master_cluster_pdf, worker_cluster_pdf])  # merge the clustering result on master and worker
    res = get_accurate(cluster_pdf, 3)
    print("方法dcd;准确率为{};Worker用时{}s;Master用时{}s".format(res, mean_worker_time, master_time))
    return [round(res, 6), round(master_time, 4), round(mean_worker_time, 4)]
