#!/home/lizhe/anaconda3/envs/pyspark/bin/python
from simulationfunc import *
from math import ceil
from pyspark import SparkContext
from pyspark import SparkConf
import os
os.environ['JAVA_HOME'] = '/usr/lib/jvm/jdk8u252'

# Spark Environment
findspark.init("/usr/local/spark")
conf = SparkConf().\
    setMaster("local[*]").\
    setAll([('spark.executor.memory', '6g'),
            ('spark.driver.memory', '10g'),
            ('num-executors', '36')])
# spark = SparkContext.getOrCreate(conf)
spark = pyspark.sql.SparkSession.builder.\
    config(conf=conf).\
    appName("Pokec").\
    getOrCreate()
#################################################

# Pandas options
pd.options.display.max_columns = None
pd.options.display.max_rows = None
np.set_printoptions(threshold=np.inf)
#################################################


# define the algorithm function of spark version
# def spark_pilot_spectral_clustering(whole_node_num, node_in_master_num, cluster_num,
#                                     worker_num, worker_per_sub_reader, adjacency_matrix, index_to_label_dict):
#     """
#     :param whole_node_num: the number of all nodes
#     :param node_in_master_num: the number of the nodes in the master
#     :param cluster_num: the number of clusers
#     :param worker_num: the number of workers
#     :param worker_per_sub_reader: the number for reading into memory per time
#     :param adjacency_matrix: the adjacency matrix
#     :param index_to_label_dict: the index2label dictionary
#     :return: mis-clustering rate
#     """
#     # Register a user defined function via the Pandas UDF
#     beta = StructType([StructField('IndexNum', IntegerType(), True),
#                        StructField('ClusterInfo', IntegerType(), True),
#                        StructField('ClusterExp', IntegerType(), True)])
#
#     @pandas_udf(beta, PandasUDFType.GROUPED_MAP)
#     def clustering_worker_udf(data_frame):
#         return clustering_worker(worker_pdf=data_frame,
#                                  master_pdf=master_pdf,
#                                  pseudo_center_dict=master_pseudo_dict,
#                                  real_data=True)
#
#     master_pdf, worker_pdf = split_master_worker(adjacency_matrix,
#                                                  index_to_label_dict,
#                                                  master_num=node_in_master_num,
#                                                  partition_num=worker_num)
#     master_pseudo_dict, master_cluster_pdf, master_time = \
#         spectral_clustering_master(master_pdf, cluster_num, real_data=True)
#     worker_size = whole_node_num - node_in_master_num
#     sub_num = ceil(worker_size / worker_per_sub_reader)
#     for i in range(sub_num):
#         if i != sub_num - 1:
#             worker_sdf_isub = spark.createDataFrame(worker_pdf[(sub_num * i): (sub_num * (i + 1) - 1)])
#         else:
#             worker_sdf_isub = spark.createDataFrame(worker_pdf[(sub_num * i): (worker_size - 1)])
#         if i == 0:
#             worker_sdf = worker_sdf_isub
#         else:
#             worker_sdf = worker_sdf.unionAll(worker_sdf_isub)
#     worker_sdf = worker_sdf.repartitionByRange("PartitionID")
#     start1 = time.time()
#     worker_cluster_sdf = worker_sdf.groupby("PartitionID").apply(clustering_worker_udf)
#     end1 = time.time()
#     worker_cluster_pdf = worker_cluster_sdf.toPandas()  # Spark DataFrame => Pandas DataFrame
#     cluster_pdf = pd.concat(
#         [master_cluster_pdf, worker_cluster_pdf])  # merge the clustering result on master and worker
#
#     mis_rate = get_accurate(cluster_pdf, cluster_num, error=True)
#     running_time = round((master_time + end1 - start1), 6)
#     return mis_rate, running_time


def spectral_clustering_master_pokec(master_pdf, cluster_number=3):
    """
    :param master_pdf: a pandas data frame contained the information about the master
    :param cluster_number: the clustering number
    :return: a pseudo dict, a pandas DataFrame about clustering result and the clustering time
    """
    master_index = master_pdf["IndexNum"].tolist()
    adjacency_matrix_master = master_pdf.iloc[:, 1:]

    # time start
    start_time = time.time()

    # first, get the laplace matrix
    laplace_matrix = get_laplace_matrix(adjacency_matrix_master,
                                        position='master',
                                        regularization=True)
    # second, get the spectral
    spectral = get_spectral(laplace_matrix, cluster_number, normalization=True, method='svd')

    # third, do k-means in spectral
    model = KMeans(n_clusters=cluster_number)
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
    for i in range(cluster_number):
        # find the minimum distance to the center point
        index = list(np.argwhere(distance_matrix[:, i] == min(distance_matrix[:, i])))
        index = int(index[0])
        pseudo_center_dict[master_index[index]] = cluster_label[index]

    # time end
    end_time = time.time()

    # return the result of the clustering on the master (a pandas DataFrame)

    out_df = pd.DataFrame(master_pdf, columns=["IndexNum"])
    out_df["ClusterExp"] = cluster_label

    running_time = end_time - start_time

    # TODO decide whether to get the error number or not
    return pseudo_center_dict, out_df, running_time


def get_adjacency_matrix(master_list, worker_list, info_dict, position='worker'):
    if position == 'master':
        assert len(master_list) == len(worker_list)
        size = len(master_list)
        adj_matrix = np.zeros((size, size), dtype=int)

        for i in range(size):
            adj_matrix[i][i] = 10
            element1 = master_list[i]
            element1_relationship_list = info_dict[element1]
            comment_element_list = list(set(element1_relationship_list) &
                                        set(master_list))
            for element2 in comment_element_list:
                adj_matrix[i][master_list.index(element2)] = 1
    else:
        m_size = len(master_list)
        w_size = len(worker_list)
        adj_matrix = np.zeros((w_size, m_size), dtype=int)

        for i in range(w_size):
            element1 = worker_list[i]
            element1_relationship_list = info_dict[element1]
            comment_element_list = list(set(element1_relationship_list) &
                                        set(master_list))
            for element2 in comment_element_list:
                adj_matrix[i][master_list.index(element2)] = 1
    return adj_matrix


# def get_relationship_of_worker_and_master(master_list, worker_list,
#                                           info_dict, partition_number=10, seed=1):
#     """
#     :param master_list: the list containing the index of the master
#     :param worker_list: the list containing the index of the worker
#     :param info_dict: the dictionary containing the relationships of each node
#     :param partition_number: the number of partitions
#     :param seed: the seed for generating the partitions
#     :return: a pandas dataframe
#     """
#     worker_len = len(worker_list)
#     prng = np.random.RandomState(seed)
#     partition_id_list = prng.randint(0, partition_number, worker_len, dtype=int)
#     index2partition_dict = dict(zip(worker_list, partition_id_list))
#     worker_adj_matrix_element_pdf = pd.DataFrame(columns=('PartitionID', 'IndexNum', 'row', 'column'))
#     for row in range(worker_len):
#         element2 = worker_list[row]
#         element1 = index2partition_dict[element2]
#         element3 = row
#
#         element1_relationship_list = info_dict[element1]
#         comment_element_list = list(set(element1_relationship_list) &
#                                     set(master_list))
#         for element4 in comment_element_list:
#             worker_adj_matrix_element_pdf = worker_adj_matrix_element_pdf.\
#                 append(pd.DataFrame({'PartitionID': [element1],
#                                      'IndexNum': [element2],
#                                      'row': [element3],
#                                      'column': [master_list.index(element4)]}),
#                        ignore_index=True)
#     return worker_adj_matrix_element_pdf


def clustering_worker_pokec(w_pdf, m_pdf, pseudo_center_dict):
    """
    :param w_pdf: the pandas data frame contained worker information
    :param m_pdf: the pandas data frame contained master information
    :param pseudo_center_dict: the result of clustering on master like {316.0: 0, 111.0: 1, 309.0: 2}
    :return: a pandas data frame (three columns, first column is the index,
    second column is the true cluster labels, third column is the cluster labels in experiment)
    """
    master_index = m_pdf["IndexNum"].tolist()
    worker_index = w_pdf["IndexNum"].tolist()
    total_index = master_index + worker_index
    pseudo_index = list(pseudo_center_dict.keys())
    adjacency_matrix_m = m_pdf.iloc[:, 1:]
    adjacency_matrix_w = w_pdf.iloc[:, 2:]
    adjacency_matrix = np.vstack((adjacency_matrix_m, adjacency_matrix_w))

    # first, get the laplace matrix
    time1 = time.time()
    laplace_matrix = get_laplace_matrix(adjacency_matrix,
                                        position='worker',
                                        regularization=True)
    # second, get the spectral
    spectral = get_spectral(laplace_matrix, len(pseudo_index), normalization=True, method='svd')

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
    time2 = time.time()
    print(round(time2 - time1, 6))

    # time end
    # end_time = time.time()

    # finally, return the pandas data frame
    out_df = pd.DataFrame(w_pdf, columns=["IndexNum"])
    out_df["ClusterExp"] = worker_cluster_list
    # pd.options.display.max_columns = None
    # pd.options.display.max_rows = None
    # print(worker_cluster_list)
    # print(out_df)

    return out_df

# def clustering_worker_pokec(w_index_list, w_pdf, m_pdf, pseudo_center_dict):
#     """
#     :param w_index_list: all the index of worker
#     :param w_pdf: the pandas data frame contained worker information
#     :param m_pdf: the pandas data frame contained master information
#     :param pseudo_center_dict: the result of clustering on master like {316.0: 0, 111.0: 1, 309.0: 2}
#     :return: a pandas data frame (three columns, first column is the index,
#     second column is the true cluster labels, third column is the cluster labels in experiment)
#     """
#     master_index = m_pdf["IndexNum"].tolist()
#     worker_index = list(set(w_pdf["IndexNum"].tolist()))  # delete the repeated values
#     total_index = master_index + worker_index
#     pseudo_index = list(pseudo_center_dict.keys())
#     adjacency_matrix_m = m_pdf.iloc[:, 1:]
#
#     # construct the worker adjacency matrix
#     w_size = len(worker_index)
#     m_size = len(master_index)
#     adjacency_matrix_w = np.zeros((w_size, m_size), dtype=int)
#     for i in range(w_size):
#         select_df = w_pdf.loc[w_pdf['IndexNum'] == worker_index[i]]
#         # row_list = select_df['row'].tolist()
#         column_list = select_df['column'].tolist()
#         for column in column_list:
#             adjacency_matrix_w[i][column] = 1
#     # adjacency_matrix_w = w_pdf.iloc[:, 2:]
#     adjacency_matrix = np.vstack((adjacency_matrix_m, adjacency_matrix_w))
#
#     # first, get the laplace matrix
#     laplace_matrix = get_laplace_matrix(adjacency_matrix,
#                                         position='worker',
#                                         regularization=True)
#     # second, get the spectral
#     spectral = get_spectral(laplace_matrix, len(pseudo_index), normalization=True, method='svd')
#
#     # third, clustering on the spectral
#     worker_cluster_list = []
#     pseudo_index_in_total_index = [total_index.index(key) for key in pseudo_index]
#     for i in range(len(total_index)):
#         # to store the distance between point mapping and each pseudo_center
#         distance_list = [np.sqrt(sum(np.power(
#             spectral[index] - spectral[i], 2)))
#             for index in pseudo_index_in_total_index]
#         if total_index[i] not in master_index:
#             worker_cluster_list.append(pseudo_center_dict[pseudo_index[distance_list.index(min(distance_list))]])
#
#     # time end
#     # end_time = time.time()
#
#     # finally, return the pandas data frame
#     # out_df = pd.DataFrame(w_pdf, columns=["IndexNum"])
#     # out_df["ClusterExp"] = worker_cluster_list
#     out_df = pd.DataFrame({"IndexNum": worker_index,
#                            "ClusterExp": worker_cluster_list})
#
#     return out_df


if __name__ == '__main__':
    partition_num = 100
    cluster_num = 3
    total_size = 1632803
    master_size_list = [12500]
    for master_size in master_size_list:
        # parameter settings
        # master_size = 100
        worker_size = total_size - master_size
        data_path = '/home/lizhe/PycharmProject/SparkSimulation/RealData/Pokec/data/'
        final_res_path = '/home/lizhe/PycharmProject/SparkSimulation/RealData/Pokec/res/'

        # Register a user defined function via the Pandas UDF
        beta = StructType([StructField('IndexNum', IntegerType(), True),
                           StructField('ClusterExp', IntegerType(), True)])


        @pandas_udf(beta, PandasUDFType.GROUPED_MAP)
        def clustering_worker_pokec_udf(data_frame):
            return clustering_worker_pokec(w_pdf=data_frame,
                                           m_pdf=master_pdf,
                                           pseudo_center_dict=master_pseudo_dict)

        # # load data
        # # print(">>>>>>>>>>>>>>>>> 加载数据 <<<<<<<<<<<<<<<<<")
        relationship_dict = np.load('data/relationship_dict_2020.8.2.npy', allow_pickle=True).item()
        sorted_index = np.load('data/degree_sorted_index.npy', allow_pickle=True).tolist()
        master_index_list = sorted_index[:master_size]  # select 'master_size' nodes
        worker_index_list = sorted_index[master_size:]
        #
        # construct the worker_adjacency_matrix
        print(">>>>>>>>>>>>>> 构造Worker邻接矩阵 <<<<<<<<<<<<<<<<")
        worker_adj_matrix_np = np.zeros((worker_size, master_size), dtype=int)
        for i in range(master_size):
            connect_list = list(set(relationship_dict[master_index_list[i]]).
                                difference(set(master_index_list)))

            for element in connect_list:
                worker_adj_matrix_np[worker_index_list.index(element)][i] = 1
            if (i+1) % 10 == 0:
                print("Running:" + str(round((i+1)/master_size, 4) * 100) + "%")
        print(">>>>>>>>>>>>>> Worker邻接矩阵构造成功 <<<<<<<<<<<<<<<<")
        partition = np.random.randint(0, partition_num, worker_size, dtype=int).reshape(worker_size, 1)
        data_worker_np = np.concatenate((partition,
                                         np.array(worker_index_list, dtype=int).reshape(worker_size, 1),
                                         worker_adj_matrix_np), 1)
        data_worker_pdf = pd.DataFrame(data_worker_np)
        data_worker_pdf.columns = ['PartitionID', 'IndexNum'] + [str(x) for x in master_index_list]
        save_file_name = data_path + 'data_worker_master_{}'.format(master_size) + '.csv'
        # print(">>>>>>>>>>>>>> 保存Worker邻接矩阵为csv文件 <<<<<<<<<<<<<<<<")
        data_worker_pdf.to_csv(save_file_name, index=False)
        # print(">>>>>>>>>>>>>> csv文件已保存 <<<<<<<<<<<<<<<<")

        # read the worker matrix
        print(">>>>>>>>>>>>>> Worker邻接矩阵读入Spark <<<<<<<<<<<<<<<<")
        master_name_list = [StructField(str(x), IntegerType(), True) for x in master_index_list]
        schema_sdf = StructType([
            StructField('PartitionID', IntegerType(), True),
            StructField('IndexNum', IntegerType(), True)] +
                                master_name_list)
        read_pdf_csv_name = 'file://' + save_file_name
        worker_sdf = spark.read.csv(read_pdf_csv_name, header=True, schema=schema_sdf)
        # print(">>>>>>>>>>>>>> 读入成功 <<<<<<<<<<<<<<<<")
        master_adjacency_matrix = \
            get_adjacency_matrix(master_index_list, master_index_list, relationship_dict, position='master')

        # master矩阵转为pandas dataframe
        print("Transform to Pandas DataFrame")
        data_master_np = np.concatenate((np.array(master_index_list, dtype=int).reshape(master_size, 1),
                                         master_adjacency_matrix), 1)
        master_pdf = pd.DataFrame(data_master_np, columns=["IndexNum"] +
                                                          [str(x) for x in master_index_list])
        # master聚类
        print("Master上聚类")
        master_pseudo_dict, master_cluster_pdf, master_time = \
            spectral_clustering_master_pokec(master_pdf, cluster_num)
        save_master_res_name = final_res_path + 'master_res_{}'.format(master_size) + '.csv'
        master_cluster_pdf.to_csv(save_master_res_name, index=False)
        master_cluster_pdf.to_json()
        print("Master聚类成功")
        print('Worker聚类开始')
        start1 = time.time()
        worker_cluster_sdf = worker_sdf.groupby("PartitionID").apply(clustering_worker_pokec_udf)
        end1 = time.time()
        print(">>>>>>>>>>>>>>>  Worker聚类结束，用时{}s  <<<<<<<<<<<<<<<".format(end1 - start1))
        print(">>>>>>>>>>>>>>> 共用时{}s <<<<<<<<<<".format(round((end1 - start1 + master_time), 4)))
        save_worker_res_file = 'file://' + final_res_path + 'worker_res_of_master_{}'.format(master_size) + '_2020.8.6'
        print(">>>>>>>>>>>>> Worker聚类结果保存为csv <<<<<<<<<<<<<<<<")
        # worker_cluster_sdf.show()
        worker_cluster_sdf.coalesce(1).write.format('json').save(save_worker_res_file)
        print(">>>>>>>>>>>>> 已保存 <<<<<<<<<<<<<<<<")
