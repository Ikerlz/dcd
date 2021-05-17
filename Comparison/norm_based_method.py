import cvxpy as cp
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark import SparkConf
import time
import itertools
from pyspark.sql.types import *
import findspark
import pyspark
from utils import *
import os


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

        sample_size = max([max(x) for x in recovered_cluster_list]) + 1

        adj_matrix = np.zeros((sample_size, sample_size))

        for row in range(self.observed_adjacency_matrix.shape[0]):
            x = self.observed_adjacency_matrix[row][0]
            y = self.observed_adjacency_matrix[row][1]
            adj_matrix[x][y] = 1
        for i in range(sample_size):
            adj_matrix[i][i] = 1

        for i in range(node_num):
            node1 = super_node_dict[i]
            for j in range(node_num):
                node2 = super_node_dict[j]
                if len(node1) + len(node2) == 2:
                    self.fused_graph_adj[i][j] = 1
                else:
                    sub_mat = adj_matrix[node1][:, node2]
                    E_hat = (np.sum(sub_mat)) / (len(node1) * len(node2))
                    if E_hat >= self.param_t:
                        self.fused_graph_adj[i][j] = 1
                    else:
                        self.fused_graph_adj[i][j] = 0
        # Construct the set of high confidence nodes
        high_confidence_nodes_list = [x for x in range(node_num) if len(clusters_list[x]) > 1]
        self.super_node_dict = super_node_dict
        self.high_confidence_nodes_list = high_confidence_nodes_list


def norm_based_clustering(args_dict, data_dict):
    # Model Settings
    pdf = data_dict["pdf"]
    total_info = data_dict["total_info"]
    cluster_num = args_dict["cluster_num"]
    cvx_t = args_dict["cvx_t"]
    cvx_T = args_dict["T"]
    cvx_C_A = args_dict["cvx_C_A"]
    cvx_C_Ac = args_dict["cvx_C_Ac"]
    cvx_C_c = args_dict["cvx_C_c"]
    split_num = args_dict["split_num"]
    # -------------------#
    params_list = [cvx_C_A, cvx_C_Ac, cvx_C_c]
    #################################################

    # Main
    # Register a user defined function via the Pandas UDF
    beta = StructType([
        StructField('PartitionID', IntegerType(), True),
        StructField('IndexNum', IntegerType(), True),
        StructField('ClusterExp', IntegerType(), True),
        StructField('Time', FloatType(), True),
    ])

    @pandas_udf(beta, PandasUDFType.GROUPED_MAP)
    def clustering_worker_udf(data_frame):
        return norm_based_worker_clustering(worker_df=data_frame, cluster_num=cluster_num)

    # construct the whole adjacency matrix
    whole_adj_mat = pdf.drop(["PartitionID"], axis=1).values
    whole_adj_mat = whole_adj_mat[whole_adj_mat[:, 1] != -1]
    node1_list = whole_adj_mat[:, 0].tolist()
    node2_list = whole_adj_mat[:, 1].tolist()
    whole_adj_mat = np.array([
        node1_list + node2_list,
        node2_list + node1_list
    ]).T

    # clustering on workers
    # print("---------- Worker Clustering ----------")
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
    sdf = spark.createDataFrame(pdf)
    worker_cluster_sdf = sdf.groupby("PartitionID").apply(clustering_worker_udf)
    # print("---------- To Pandas ----------")
    worker_cluster_pdf = worker_cluster_sdf.toPandas()
    t_worker = np.mean(list(set(worker_cluster_pdf["Time"].values.tolist())))
    worker_cluster_pdf = worker_cluster_pdf.drop(columns=["Time"])
    # clustering on master
    # print("---------- Master Clustering ----------")
    time2 = time.time()
    clustering_master = FusedGraphClustering(observed_adjacency_matrix=whole_adj_mat,
                                             cluster_number=cluster_num,
                                             param_l=split_num,
                                             param_t=cvx_t,
                                             param_T=cvx_T,
                                             params_for_norm_based_clustering=params_list)

    worker_clustering_clusters_list = \
        list(
            worker_cluster_pdf.groupby(by=["PartitionID", "ClusterExp"]).apply(lambda x: x["IndexNum"].tolist()).values)
    # print("---------- Build Fused Graph ----------")
    clustering_master.build_fused_graph(worker_clustering_clusters_list)
    # print("---------- Solve Convex Problem ----------")
    clustering_master.norm_based_clustering()
    # print("---------- Spectral Clustering ----------")
    res = clustering_master.spectral_clustering()
    time3 = time.time()
    res_span = [[] for _ in range(cluster_num)]
    for i in range(cluster_num):
        for item in res[i]:
            res_span[i] = res_span[i] + item

    # calculate the accuracy
    real_label = [[] for _ in range(cluster_num)]

    for row in worker_cluster_pdf.itertuples(index=False, name='Pandas'):
        item = getattr(row, "IndexNum")
        label = total_info[item]
        real_label[label].append(item)

    accuracy_matrix = np.zeros((cluster_num, cluster_num))
    for i in range(cluster_num):
        for j in range(cluster_num):
            accuracy_matrix[i][j] = len(set(real_label[i]).intersection(set(res_span[j])))
    case_iterator = itertools.permutations(range(cluster_num), cluster_num)

    accurate = 0
    for item in case_iterator:
        acc = sum([accuracy_matrix[i][item[i]] for i in range(cluster_num)])
        if acc > accurate:
            accurate = acc
    res = accurate / np.sum(accuracy_matrix)
    t_master = round(time3 - time2, 4)
    print("方法norm_based;准确率为{};Worker用时{}s;Master用时{}s".format(res, t_worker, t_master))
    return [round(res, 6), round(t_master, 4), round(t_worker, 4)]
