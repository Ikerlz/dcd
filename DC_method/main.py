from DC_method.util import *
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
sbm_matrix = sbm_matrix4
sample_size = 1000
# master_num = 1000
# worker_per_sub = 2000
partition_num = 10
cluster_num = 3
#################################################

# Main
# Register a user defined function via the Pandas UDF
beta = StructType([StructField('IndexNum', IntegerType(), True),
                   # StructField('ClusterInfo', IntegerType(), True),
                   StructField('ClusterExp', IntegerType(), True)])


@pandas_udf(beta, PandasUDFType.GROUPED_MAP)
def clustering_worker_udf(data_frame):
    return worker_clustering(worker_df=data_frame,
                             cluster_num=cluster_num)


pdf, total_info = simulate_sbm_dc_data(sbm_matrix)
sdf = spark.createDataFrame(pdf)
worker_cluster_sdf = sdf.groupby("PartitionID").apply(clustering_worker_udf)
worker_cluster_pdf = worker_cluster_sdf.toPandas()
real_label = []
for row in worker_cluster_pdf.itertuples(index=False, name='Pandas'):
    item = getattr(row, "IndexNum")
    real_label.append(total_info[item])
worker_cluster_pdf["ClusterInfo"] = real_label
print(get_accurate(worker_cluster_pdf, 3))
print(worker_cluster_pdf)

