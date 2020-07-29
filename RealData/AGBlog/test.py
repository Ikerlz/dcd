from simulationfunc import *
from scipy.io import loadmat
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
            ('spark.driver.memory', '4g')])
# spark = SparkContext.getOrCreate(conf)
spark = pyspark.sql.SparkSession.builder.\
    config(conf=conf).\
    appName("AGBlog").\
    getOrCreate()
#################################################


def spark_pilot_spectral_clustering(whole_node_num, node_in_master_num, cluster_num,
                                    worker_num, worker_per_sub, adjacency_matrix, index_to_label_dict):
    """
    :param whole_node_num: the number of all nodes
    :param node_in_master_num: the number of the nodes in the master
    :param cluster_num: the number of clusers
    :param worker_num: the number of workers
    :param worker_per_sub: the number for reading into memory per time
    :param adjacency_matrix: the adjacency matrix
    :param index_to_label_dict: the index2label dictionary
    :return: mis-clustering rate
    """
    # Register a user defined function via the Pandas UDF
    beta = StructType([StructField('IndexNum', IntegerType(), True),
                       StructField('ClusterInfo', IntegerType(), True),
                       StructField('ClusterExp', IntegerType(), True)])

    @pandas_udf(beta, PandasUDFType.GROUPED_MAP)
    def clustering_worker_udf(data_frame):
        return clustering_worker(worker_pdf=data_frame,
                                 master_pdf=master_pdf,
                                 pseudo_center_dict=master_pseudo_dict,
                                 real_data=True)

    master_pdf, worker_pdf = split_master_worker(adjacency_matrix,
                                                 index_to_label_dict,
                                                 master_num=node_in_master_num,
                                                 partition_num=worker_num)
    master_pseudo_dict, master_cluster_pdf, master_time = \
        spectral_clustering_master(master_pdf, cluster_num, real_data=True)
    worker_size = whole_node_num - node_in_master_num
    sub_num = ceil(worker_size / worker_per_sub)
    for i in range(sub_num):
        if i != sub_num - 1:
            worker_sdf_isub = spark.createDataFrame(worker_pdf[(sub_num * i): (sub_num * (i + 1) - 1)])
        else:
            worker_sdf_isub = spark.createDataFrame(worker_pdf[(sub_num * i): (worker_size - 1)])
        if i == 0:
            worker_sdf = worker_sdf_isub
        else:
            worker_sdf = worker_sdf.unionAll(worker_sdf_isub)
    worker_sdf = worker_sdf.repartitionByRange("PartitionID")
    start1 = time.time()
    worker_cluster_sdf = worker_sdf.groupby("PartitionID").apply(clustering_worker_udf)
    end1 = time.time()
    worker_cluster_pdf = worker_cluster_sdf.toPandas()  # Spark DataFrame => Pandas DataFrame
    cluster_pdf = pd.concat(
        [master_cluster_pdf, worker_cluster_pdf])  # merge the clustering result on master and worker

    mis_rate = get_accurate(cluster_pdf, cluster_num, error=True)
    running_time = round((master_time + end1 - start1), 6)
    return mis_rate, running_time
