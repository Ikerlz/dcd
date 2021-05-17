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
    appName("SparkSC").\
    getOrCreate()
#################################################

# Pandas options
pd.options.display.max_columns = None
pd.options.display.max_rows = None
np.set_printoptions(threshold=np.inf)
#################################################

# Model Settings
p = 0.2
_p = p / 2
sbm_matrix = np.array([[p, _p, _p], [_p, p, _p], [_p, _p, p]])
sample_size = 10000
master_num = 1000
worker_per_sub = 900
partition_num = 10
cluster_num = 3
#################################################

# Main
# Register a user defined function via the Pandas UDF
beta = StructType([StructField('IndexNum', IntegerType(), True),
                   StructField('ClusterInfo', IntegerType(), True),
                   StructField('ClusterExp', IntegerType(), True)])


@pandas_udf(beta, PandasUDFType.GROUPED_MAP)
def clustering_worker_udf(data_frame):
    return clustering_worker(worker_pdf=data_frame,
                             master_pdf=master_pdf,
                             pseudo_center_dict=master_pseudo_dict)
# start0 = time.time()
# print("simulate data")


print(">>>>>>>>>>>>>>>  开始生成模拟数据  <<<<<<<<<<<<<<<")
start0 = time.time()
master_pdf, worker_pdf = simulate_sbm_data(sbm_matrix,
                                           sample_size,
                                           master_num,
                                           partition_num,
                                           cluster_num)
end0 = time.time()
print(">>>>>>>>>>>>>>>  生成模拟数据完成，用时{}s  <<<<<<<<<<<<<<<".format(str(end0 - start0)))
# print("start")
print(">>>>>>>>>>>>>>>  Master聚类开始  <<<<<<<<<<<<<<<")
master_pseudo_dict, master_cluster_pdf, master_time = spectral_clustering_master(master_pdf, cluster_num)
print(">>>>>>>>>>>>>>>  Master聚类结束，用时{}s  <<<<<<<<<<<<<<<".format(master_time))
worker_size = sample_size - master_num
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
print(">>>>>>>>>>>>>>>  Spark DataFrame => Pandas DataFrame  <<<<<<<<<<<<<<<")
worker_cluster_pdf = worker_cluster_sdf.toPandas()  # Spark DataFrame => Pandas DataFrame
end1 = time.time()
print(">>>>>>>>>>>>>>>  Worker聚类结束，用时{}s  <<<<<<<<<<<<<<<".format(end1 - start1))
cluster_pdf = pd.concat([master_cluster_pdf, worker_cluster_pdf])  # merge the clustering result on master and worker

print(get_accurate(cluster_pdf, 3))

