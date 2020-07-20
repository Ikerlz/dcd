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
            ('spark.driver.memory', '4g')])
# spark = SparkContext.getOrCreate(conf)
spark = pyspark.sql.SparkSession.builder.\
    config(conf=conf).\
    appName("Pumbed").\
    getOrCreate()
#################################################

# Pandas options
pd.options.display.max_columns = None
pd.options.display.max_rows = None
np.set_printoptions(threshold=np.inf)
#################################################

if __name__ == '__main__':
    node_df = pd.read_csv('node_index_label.csv')
    node_relationship = pd.read_csv('node_relationship.csv')
    index_list = list(node_df.iloc[:, 2])
    label_list = list(node_df.iloc[:, 1])
    label_list = [(x-1) for x in label_list]  # let the label index starts from zero
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

    # settings
    total_size = 19717
    pilot_node_number = 2000
    cluster_number = 3
    worker_number = 2  # equal to the partition number
    worker_per_sub = 2000

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


    master_pdf, worker_pdf = split_master_worker(pumbed_adjacency_matrix, index2label_dict,
                                                 master_num=pilot_node_number, partition_num=worker_number)

    print(">>>>>>>>>>>>>>>  Master聚类开始  <<<<<<<<<<<<<<<")
    master_pseudo_dict, master_cluster_pdf, master_time = \
        spectral_clustering_master(master_pdf, cluster_number, real_data=True)
    print(">>>>>>>>>>>>>>>  Master聚类结束，用时{}s  <<<<<<<<<<<<<<<".format(master_time))
    worker_size = total_size - pilot_node_number
    sub_num = ceil(worker_size / worker_per_sub)
    print(">>>>>>>>>>>>>>>  Pandas DataFrame => Spark DataFrame  <<<<<<<<<<<<<<<")
    for i in range(sub_num):
        if i != sub_num - 1:
            worker_sdf_isub = spark.createDataFrame(worker_pdf[(sub_num * i): (sub_num * (i + 1) - 1)])
        else:
            worker_sdf_isub = spark.createDataFrame(worker_pdf[(sub_num * i): (worker_size - 1)])
        if i == 0:
            worker_sdf = worker_sdf_isub
        else:
            worker_sdf = worker_sdf.unionAll(worker_sdf_isub)
    # worker_sdf = spark.createDataFrame(worker_pdf)  # Convert Pandas DataFrame to Spark DataFrame
    worker_sdf = worker_sdf.repartitionByRange("PartitionID")
    # worker_sdf = worker_sdf.repartition(partition_num, "PartitionID")
    # print("worker")
    # start1 = time.time()
    print(">>>>>>>>>>>>>>>  Worker聚类开始  <<<<<<<<<<<<<<<")
    start1 = time.time()
    worker_cluster_sdf = worker_sdf.groupby("PartitionID").apply(clustering_worker_udf)
    end1 = time.time()
    print(">>>>>>>>>>>>>>>  Worker聚类结束，用时{}s  <<<<<<<<<<<<<<<".format(end1 - start1))
    print(">>>>>>>>>>>>>>>  Spark DataFrame => Pandas DataFrame  <<<<<<<<<<<<<<<")
    worker_cluster_pdf = worker_cluster_sdf.toPandas()  # Spark DataFrame => Pandas DataFrame
    cluster_pdf = pd.concat(
        [master_cluster_pdf, worker_cluster_pdf])  # merge the clustering result on master and worker

    print(get_accurate(cluster_pdf, cluster_number))
