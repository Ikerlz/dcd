#!/home/lizhe/anaconda3/envs/pyspark/bin/python
from simulationfunc import *
from math import ceil
from pyspark import SparkContext
from pyspark import SparkConf
import os
import itertools
os.environ['JAVA_HOME'] = '/usr/lib/jvm/jdk8u252'

# Spark Environment
findspark.init("/usr/local/spark")
conf = SparkConf().\
    setMaster("local[*]").\
    setAll([('spark.executor.memory', '6g'),
            ('spark.driver.memory', '10g'),
            ('num-executors', 36),
            ('spark.default.parallelism', 50),
            ('spark.locality.wait', '10min')])
# spark = SparkContext.getOrCreate(conf)
spark = pyspark.sql.SparkSession.builder.\
    config(conf=conf).\
    appName("Pokec").\
    getOrCreate()
sc=SparkContext.getOrCreate()
#################################################

# Pandas options
pd.options.display.max_columns = None
pd.options.display.max_rows = None
np.set_printoptions(threshold=np.inf)
#################################################


def calculate_relative_density(relate_dict: dict, result_dict: dict):
    """
    :param relate_dict: the dictionary recording the relationships between the nodes
    :param result_dict: the final result of the clustering
    :return: the relative density
    """
    between_denominator = 0
    between_numerator = 0
    within_denominator = 0
    within_numerator = 0

    total_node_list = list(itertools.chain(*list(result_dict.values())))

    key_list = result_dict.keys()
    for key in key_list:
        within_list = result_dict[key]
        between_list = list(set(total_node_list).difference(set(within_list)))
        within_list_len = len(within_list)
        between_list_len = len(between_list)
        within_denominator += within_list_len * (within_list_len - 1)
        between_denominator += within_list_len * between_list_len

        for item in within_list:
            all_connection = relate_dict[item]
            within_numerator += len(set(within_list) & set(all_connection))
            between_numerator += len(set(between_list) & set(all_connection))
    return between_numerator / between_denominator * within_denominator / within_numerator


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


def clustering_worker_pokec(w_pdf, m_pdf, pseudo_center_dict):
    """
    :param w_pdf: the pandas data frame contained worker information
    :param m_pdf: the pandas data frame contained master information
    :param pseudo_center_dict: the result of clustering on master like {316.0: 0, 111.0: 1, 309.0: 2}
    :return: a pandas data frame (three columns, first column is the index,
    second column is the true cluster labels, third column is the cluster labels in experiment)
    """
    # Test Broadcast
    m_pdf = m_pdf.value
    pseudo_center_dict = pseudo_center_dict.value
    master_index = m_pdf["IndexNum"].tolist()
    worker_index = list(set(w_pdf["Row"].tolist()))
    total_index = master_index + worker_index
    w_len = len(worker_index)
    m_len = len(master_index)
    adjacency_matrix_w = np.zeros((w_len, m_len), dtype=int)
    for i in range(w_len):
        row = worker_index[i]
        row_connected_list = w_pdf[w_pdf['Row'].isin([row])]['Column'].tolist()
        row_connected_to_master_list = list(set(row_connected_list) & set(master_index))
        if row_connected_to_master_list:
            for column in row_connected_to_master_list:
                adjacency_matrix_w[i][master_index.index(column)] = 1
    pseudo_index = list(pseudo_center_dict.keys())
    adjacency_matrix_m = m_pdf.iloc[:, 1:]
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

    # calculate the relative density
    # res_dict_key = [x for x in range(len(pseudo_index))]
    # res_dict_value = [[] for x in range(len(pseudo_index))]
    # res_dict = dict(zip(res_dict_key, res_dict_value))
    # pseudo_index_in_total_index = [total_index.index(key) for key in pseudo_index]
    # for i in range(len(total_index)):
    #     distance_list = [np.sqrt(sum(np.power(
    #             spectral[index] - spectral[i], 2)))
    #             for index in pseudo_index_in_total_index]
    #     if total_index[i] not in master_index:
    #         res_dict[pseudo_center_dict[pseudo_index[distance_list.index(min(distance_list))]]].append(total_index[i])
    # time2 = time.time()
    # print(round(time2 - time1, 6))
    # relative_density = calculate_relative_density(relate_dict, res_dict)

    # finally, return the pandas data frame
    out_df = pd.DataFrame({"IndexNum": worker_index,
                           "ClusterExp": worker_cluster_list})
    # out_df = pd.DataFrame({"RelativeDensity": [round(relative_density, 5)]})

    return out_df


if __name__ == '__main__':
    partition_num = 500
    cluster_num = 3
    total_size = 1632803
    master_size_list = [20000]
    data_path = '/home/lizhe/PycharmProject/SparkSimulation/RealData/Pokec/data/'
    final_res_path = '/home/lizhe/PycharmProject/SparkSimulation/RealData/Pokec/res/'
    for master_size in master_size_list:

        print('>>>>>>>>>> Register a User Defined Function <<<<<<<<<<')
        # Register a user defined function via the Pandas UDF
        beta = StructType([StructField('IndexNum', IntegerType(), True),
                           StructField('ClusterExp', IntegerType(), True)])
        # beta = StructType([StructField('RelativeDensity', FloatType(), True)])

        # @pandas_udf(beta, PandasUDFType.GROUPED_MAP)
        # def clustering_worker_pokec_udf(data_frame):
        #     return clustering_worker_pokec(w_pdf=data_frame,
        #                                    m_pdf=master_pdf,
        #                                    pseudo_center_dict=master_pseudo_dict)

        # Test Broadcast
        @pandas_udf(beta, PandasUDFType.GROUPED_MAP)
        def clustering_worker_pokec_udf(data_frame):
            return clustering_worker_pokec(w_pdf=data_frame,
                                           m_pdf=master_pdf_broadcast,
                                           pseudo_center_dict=master_pseudo_dict_broadcast)

        print('>>>>>>>>>> Select the pilot nodes and construct the adjacency matrix in master<<<<<<<<<<')
        relationship_dict = np.load('data/relationship_dict_2020.8.2.npy', allow_pickle=True).item()
        sorted_index = np.load('data/degree_sorted_index.npy', allow_pickle=True).tolist()
        master_index_list = sorted_index[:master_size]  # select 'master_size' nodes
        worker_index_list = sorted_index[master_size:]

        print('>>>>>>>>>> Construct the adjacency matrix in worker <<<<<<<<<<')

        column_list = list(relationship_dict.values())
        row_list = list(relationship_dict.keys())
        partition_list = np.random.randint(0, partition_num, total_size, dtype=int).tolist()
        row2partition_dict = dict(zip(row_list, partition_list))
        row_index_list = [[row_list[x-1]]*len(column_list[x-1]) for x in range(1, 1632804)]
        column_index_list = list(itertools.chain(*column_list))
        partition_id_list = [[row2partition_dict[row_list[x-1]]]*len(column_list[x-1]) for x in range(1, 1632804)]
        row_index_list = list(itertools.chain(*row_index_list))
        partition_id_list = list(itertools.chain(*partition_id_list))

        pdf = pd.DataFrame({'PartitionID': partition_id_list,
                            'Row': row_index_list,
                            'Column': column_index_list})

        # delete the nodes of master
        worker_pdf = pdf[~pdf['Row'].isin(master_index_list)]

        # save the worker pdf as 'json' file
        save_worker_pdf_path = data_path + 'worker_pdf_master{}.json'.format(master_size)
        print('>>>>>>>>>> Save the worker pdf <<<<<<<<<<')
        worker_pdf.to_json(save_worker_pdf_path, orient='records', lines=True)
        print('>>>>>>>>>> PDF saved <<<<<<<<<<')

        # transform pdf to sdf
        print(">>>>>>>>>>>>>> Worker pdf => Spark dataframe <<<<<<<<<<<<<<<<")
        master_name_list = [StructField(str(x), IntegerType(), True) for x in master_index_list]
        schema_sdf = StructType([
            StructField('PartitionID', IntegerType(), True),
            StructField('Row', IntegerType(), True),
            StructField('Column', IntegerType(), True)
        ])
        read_pdf_json_name = 'file://' + save_worker_pdf_path
        worker_sdf = spark.read.json(read_pdf_json_name, schema=schema_sdf)
        print(">>>>>>>>>>>>>> Pandas DataFrame has transformed to Spark DataFrame<<<<<<<<<<<<<<<<")

        # clustering on master
        print(">>>>>>>>>> Clustering on master <<<<<<<<<<")
        # construct the adjacency matrix
        master_adjacency_matrix = \
            get_adjacency_matrix(master_index_list, master_index_list, relationship_dict, position='master')
        # master矩阵转为pandas dataframe
        data_master_np = np.concatenate((np.array(master_index_list, dtype=int).reshape(master_size, 1),
                                         master_adjacency_matrix), 1)
        master_pdf = pd.DataFrame(data_master_np, columns=["IndexNum"] +
                                                          [str(x) for x in master_index_list])

        master_pseudo_dict, master_cluster_pdf, master_time = \
            spectral_clustering_master_pokec(master_pdf, cluster_num)

        # Test Broadcast
        master_pdf_broadcast = sc.broadcast(master_pdf)
        master_pseudo_dict_broadcast = sc.broadcast(master_pseudo_dict)
        #################

        save_master_res_name = final_res_path + 'master_res_{}'.format(master_size) + '.csv'
        master_cluster_pdf.to_csv(save_master_res_name, index=False)
        print(">>>>>>>>>> Finished! <<<<<<<<<<")

        # clustering on worker
        print(">>>>>>>>>> Clustering on worker <<<<<<<<<<")
        worker_cluster_sdf = worker_sdf.groupby("PartitionID").apply(clustering_worker_pokec_udf)
        print(">>>>>>>>>> Start! <<<<<<<<<<")
        # save_worker_res_file = 'file://' + final_res_path + 'worker_res_of_master_{}'.format(master_size)
        save_worker_res_file = 'file://' + final_res_path + 'worker_res_of_master_{}'.format(master_size) + '.json'
        worker_cluster_pdf = worker_cluster_sdf.to_Pandas()
        worker_cluster_pdf.to_json(save_worker_res_file,)
        # worker_cluster_sdf.coalesce(1).write.format('json').save(save_worker_res_file)
        print(">>>>>>>>>> Finished! <<<<<<<<<<")
