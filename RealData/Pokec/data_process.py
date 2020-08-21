import pandas as pd     # 引入pandas包
import numpy as np

pokec_data = pd.read_table('data/soc-pokec-relationships.txt', sep='\t', header=None)
pokec_data.columns = ['A', 'B']
# 读入txt文件，分隔符为\t
# print(pokec_data)  # 30622564行

# 共30622564行

row_count = 30622564
node_count = 1632803

relationship_dict = dict()

relationship_num = 0

for node in range(1, node_count+1):
    relationship_dict[node] = []

count = 0
for row in pokec_data.itertuples():
    item1 = getattr(row, 'A')
    item2 = getattr(row, 'B')
    relationship_dict[item1].append(item2)
    relationship_dict[item2].append(item1)
    count += 1
    if not count % 10000:
        print(count)
# for node in range(1, node_count+1):

    # list1_connect_node = pokec_data.loc[pokec_data[0] == node][1].tolist()
    # list2_connect_node = pokec_data.loc[pokec_data[1] == node][0].tolist()

    # 求两个列表的共同元素
    # list_connect_node = list(set(list1_connect_node) | set(list2_connect_node))

    # relationship_dict[node] = list1_connect_node + list2_connect_node
    # relationship_num += len(relationship_dict)

    # 进度表

    # print(node)
    # if node % 1000 == 0:
    #     print(str(round(node / row_count, 4) * 100)+'%')

# print(relationship_num)

# Save
np.save('data/relationship_dict.npy', relationship_dict)

# Load
# read_dictionary = np.load('my_file.npy').item()
# print(read_dictionary['hello'])  # displays "world"
