#!/home/lizhe/anaconda3/envs/pyspark/bin/python
from simulationfunc import *
from scipy.io import loadmat
from math import ceil
from pyspark import SparkContext
from pyspark import SparkConf
import os
import warnings

warnings.filterwarnings('ignore')
os.environ['JAVA_HOME'] = '/usr/lib/jvm/jdk8u252'

# Spark Environment
findspark.init("/usr/local/spark")
conf = SparkConf().\
    setMaster("local[*]").\
    setAll([('spark.executor.memory', '6g'),
            ('spark.driver.memory', '10g')])
# spark = SparkContext.getOrCreate(conf)
spark = pyspark.sql.SparkSession.builder.\
    config(conf=conf).\
    appName("Pubmed").\
    getOrCreate()
#################################################

# Pandas options
pd.options.display.max_columns = None
pd.options.display.max_rows = None
np.set_printoptions(threshold=np.inf)
#################################################


# define the algorithm function of spark version
def spark_pilot_spectral_clustering(whole_node_num, node_in_master_num, cluster_num,
                                    worker_num, worker_per_sub_reader, adjacency_matrix, index_to_label_dict):
    """
    :param whole_node_num: the number of all nodes
    :param node_in_master_num: the number of the nodes in the master
    :param cluster_num: the number of clusers
    :param worker_num: the number of workers
    :param worker_per_sub_reader: the number for reading into memory per time
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
    print(master_time)
    worker_size = whole_node_num - node_in_master_num
    sub_num = ceil(worker_size / worker_per_sub_reader)
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
    # start1 = time.time()
    worker_cluster_sdf = worker_sdf.groupby("PartitionID").apply(clustering_worker_udf)
    # end1 = time.time()
    worker_cluster_pdf = worker_cluster_sdf.toPandas()  # Spark DataFrame => Pandas DataFrame
    cluster_pdf = pd.concat(
        [master_cluster_pdf, worker_cluster_pdf])  # merge the clustering result on master and worker

    mis_rate = get_accurate(cluster_pdf, cluster_num, error=True)
    running_time = round(master_time, 6)
    return mis_rate, running_time


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
    print('构造矩阵')
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
    # pilot_node_number = 400
    pilot_ratio_list = [0.02*x for x in range(13, 14)]
    cluster_number = 3
    worker_number = 2  # equal to the partition number
    worker_per_sub = 4000
    repeat_number = 1
    print('开始聚类')
    for pilot_ratio in pilot_ratio_list:
        pilot_node_number = math.ceil(pilot_ratio * total_size)
        mis_rate_list = []
        running_time_list = []
        for repeat in range(repeat_number):
            mis_rate_i, running_time_i = \
                spark_pilot_spectral_clustering(total_size, pilot_node_number,
                                                cluster_number, worker_number,
                                                worker_per_sub, pumbed_adjacency_matrix, index2label_dict)
            mis_rate_list.append(mis_rate_i)
            running_time_list.append(running_time_i)
            print('一次运行结束')
        print('r:{},R:{},time:{}'.format(pilot_ratio,
                                         round(sum(mis_rate_list)/len(mis_rate_list), 5),
                                         round(sum(running_time_list)/len(running_time_list), 5)
                                         )
              )
