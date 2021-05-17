#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：dcd 
@File    ：main.py
@Author  ：Iker Zhe
@Date    ：2021/3/5 16:13 
'''

from norm_based_method import *
from dcd_method import dcd_clustering
###############################################
# Pandas options
pd.options.display.max_columns = None
pd.options.display.max_rows = None
np.set_printoptions(threshold=np.inf)
#################################################

if __name__ == '__main__':
    # ---------- 测试worker对运行时间的影响 ---------- #
    p_list = [0.2, 0.6]
    repeat_num = 1
    n = 10000
    # worker_num = 10
    worker_num_list = [10*x for x in range(1, 6)]
    cluster_num = 3
    K = 5
    c0 = 1  # a constant
    lam = c0 / np.sqrt(n * np.log(n))
    C_c = c0 / np.sqrt(K * np.log(n))
    # other constants
    T = 10  # a constant
    l = 1  # a constant
    master_num = int(0.1 * n)
    settings_dict = dict()
    data_dict = dict()
    settings_dict["sample_size"] = n
    settings_dict["master_num"] = master_num
    settings_dict["worker_per_sub"] = 1000
    settings_dict["cluster_num"] = cluster_num
    settings_dict["cvx_C_c"] = C_c
    settings_dict["T"] = T
    settings_dict["split_num"] = l
    # 储存结果
    res_dcd = np.zeros((100, 3))
    res_norm = np.zeros((100, 3))
    count = 0
    for p in p_list:
        _p = 0.5 * p
        t = 0.5 * p * 1.1
        C_A = np.sqrt((1 - t) / t) * lam
        C_Ac = np.sqrt(t / (1 - t)) * lam
        sbm_matrix = np.array([[p, _p, _p], [_p, p, _p], [_p, _p, p]])
        settings_dict["cvx_t"] = t
        settings_dict["cvx_C_A"] = C_A
        settings_dict["cvx_C_Ac"] = C_Ac
        for worker_num in worker_num_list:
            print("----------- Generate Data ----------")
            for r in range(repeat_num):
                pdf, total_info, dcd_master_pdf, dcd_worker_pdf = \
                    simulate_sbm_dc_data(sbm_matrix, sample_size=n, dcd_master_node=master_num,
                                         partition_num=worker_num, method="both")
                data_dict_norm = {"pdf": pdf, "total_info": total_info}
                data_dict_dcd = {"master_pdf": dcd_master_pdf, "worker_pdf": dcd_worker_pdf}
                res_norm[count] = norm_based_clustering(settings_dict, data_dict_norm)
                res_dcd[count] = dcd_clustering(settings_dict, data_dict_dcd)
                count += 1
                print("========== M={}; 进度：{}% ==========".format(worker_num, count))




    # -----------------------------------
    # p_list = [0.05*x for x in range(2, 16)]
    # n_list = [1000*x for x in range(1, 11)]
    # repeat_num = 10
    # res_dcd = np.zeros((10, 3))
    # res_norm = np.zeros((10, 3))
    # for term in range(len(n_list)):
    #     temp_mat_dcd = np.zeros((repeat_num, 3))
    #     temp_mat_norm = np.zeros((repeat_num, 3))
    #     for r in range(repeat_num):
    #         n = n_list[term]
    #         # settings
    #         settings_dict = dict()
    #         data_dict = dict()
    #         # for SBM matrix
    #         p = 0.6
    #         _p = 0.5 * p
    #         sbm_matrix = np.array([[p, _p, _p], [_p, p, _p], [_p, _p, p]])
    #         # sample size
    #         # n = 10000
    #         settings_dict["sample_size"] = n
    #         # master number
    #         master_num = int(0.1 * n)
    #         settings_dict["master_num"] = master_num
    #         # worker per subset
    #         settings_dict["worker_per_sub"] = 1000
    #         # worker number
    #         worker_num = 10
    #         # cluster number
    #         cluster_num = 3
    #         settings_dict["cluster_num"] = cluster_num
    #         # CVXPY PARAMETERS
    #         K = 5  # K is the size of the smallest cluster
    #         c0 = 1  # a constant
    #         lam = c0 / np.sqrt(n * np.log(n))
    #         t = 0.5 * p * 1.1
    #         C_A = np.sqrt((1 - t) / t) * lam
    #         C_Ac = np.sqrt(t / (1 - t)) * lam
    #         C_c = c0 / np.sqrt(K * np.log(n))
    #         settings_dict["cvx_t"] = t
    #         settings_dict["cvx_C_A"] = C_A
    #         settings_dict["cvx_C_Ac"] = C_Ac
    #         settings_dict["cvx_C_c"] = C_c
    #         # other constants
    #         T = 10  # a constant
    #         l = 1  # a constant
    #         settings_dict["T"] = T
    #         settings_dict["split_num"] = l
    #
    #         # generate data
    #         print("----------- Generate Data ----------")
    #         pdf, total_info, dcd_master_pdf, dcd_worker_pdf = \
    #             simulate_sbm_dc_data(sbm_matrix, sample_size=n, dcd_master_node=master_num, partition_num=worker_num, method="both")
    #         data_dict_norm = {"pdf": pdf, "total_info": total_info}
    #         data_dict_dcd = {"master_pdf": dcd_master_pdf, "worker_pdf": dcd_worker_pdf}
    #         temp_mat_norm[r] = norm_based_clustering(settings_dict, data_dict_norm)
    #         temp_mat_dcd[r] = dcd_clustering(settings_dict, data_dict_dcd)
    #         print("========== 进度：{}% ==========".format((r+1)*(term+1)*100/(len(n_list)*repeat_num)))
    #
    #     res_dcd[term] = np.mean(temp_mat_dcd, axis=0)
    #     res_norm[term] = np.mean(temp_mat_norm, axis=0)









