from DC_method.util import *
from DC_method.cvtest import *
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
beta = StructType([
    StructField('PartitionID', IntegerType(), True),
    StructField('IndexNum', IntegerType(), True),
    StructField('ClusterExp', IntegerType(), True)
])


@pandas_udf(beta, PandasUDFType.GROUPED_MAP)
def clustering_worker_udf(data_frame):
    return worker_clustering(worker_df=data_frame,
                             cluster_num=cluster_num)


pdf, total_info = simulate_sbm_dc_data(sbm_matrix)
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
sdf = spark.createDataFrame(pdf)
worker_cluster_sdf = sdf.groupby("PartitionID").apply(clustering_worker_udf)
worker_cluster_pdf = worker_cluster_sdf.toPandas()

# 构造fused graph
worker_clustering_clusters_list = []
worker_cluster_npy = worker_cluster_pdf.values
for i in range(partition_num):
    sub_mat = worker_cluster_npy[worker_cluster_npy[:, 0] == i]
    for j in range(cluster_num):
        l = sub_mat[sub_mat[:, 2] == j][:, 1].tolist()
        worker_clustering_clusters_list.append(l)
fused_graph_adj_mat, super_node_dict, high_confidence_nodes_list = build_fused_graph(observed_adj_mat=whole_adj_mat,
                                                                                     clusters_list=worker_clustering_clusters_list,
                                                                                     l=4, t=0.6, T=20)


# master上聚类
construct_A_set = []
node_num = len(fused_graph_adj_mat)
for i in range(node_num):
    for j in range(node_num):
        if fused_graph_adj_mat[i][j]:
            construct_A_set.append((i, j))
construct_C_set = []
for i in range(len(high_confidence_nodes_list)):
    for j in range(len(high_confidence_nodes_list)):
        construct_C_set.append((i, j))

solve_fused_graph(total_node_num=node_num,
                  param_A=1,
                  param_Ac=1,
                  param_C=1,
                  A_set=construct_A_set,
                  C_set=construct_C_set,
                  adj_matrix=fused_graph_adj_mat)



# real_label = []
# for row in worker_cluster_pdf.itertuples(index=False, name='Pandas'):
#     item = getattr(row, "IndexNum")
#     real_label.append(total_info[item])
# worker_cluster_pdf["ClusterInfo"] = real_label
# print(get_accurate(worker_cluster_pdf, 3))
# print(worker_cluster_pdf)

