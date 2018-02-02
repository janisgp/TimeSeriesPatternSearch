'''
Created on 23 Jan 2018

@author: Janis
'''

import FeatureExtraction as FE
import numpy as np
import LoadData as LD
from matplotlib import pyplot as plt
import Tree
#import test_running_mean as trm

############################################################################
####################### HYPERPARAMETERS ####################################
############################################################################
time_granularity = 2            # -> resembles n + (n-1) time intervals
cluster_granularity = 0.008     # -> controls amount of clusters (% of scenario count)
depth = 2                       # -> depth of tree
interval_of_interest=[0,10]     # -> quantile of input data that resembles the target in %
#window = 5                     # -> sliding window for running mean
############################################################################
############################################################################
############################################################################

############################################################################
####################### LOADING THE DATA ###################################
############################################################################
################### OW ###################
scenarios_path = "OW-2017Q4_Scenarios_withCS.csv"
results_path = "OW-2017Q4_Results.csv"
inp, out, var_names = LD.load_data_to_numpy(scenarios_path, results_path,
                                            result_col_name="PVSP_CS",result_oneIndex=True,dec_sep=';')
#inp = trm.running_mean_var(inp,window)      # -> compute running mean of time series
################## AZL ###################
# scenarios_path = "DatenAllianz/AZL-2017Q4_Scenarios_merged_with_cat.csv"
# results_path = "DatenAllianz/AZL-2017Q4_Results.csv"
# inp, out, var_names = LD.load_data_to_numpy(scenarios_path, results_path,
#                                             result_col_name="PVSP",result_oneIndex=True,dec_sep=',')

targets = LD.identify_targets(out,interval_of_interest)

inp_targets = inp[:,targets]
############################################################################
############################################################################
############################################################################

############################################################################
########################### CLUSTERING #####################################
############################################################################
cluster_dict = FE.cluster_hierarchy(inp_targets[0],cluster_granularity=0.05)
patterns, patterns_dicts, patterns_names, patterns_var = FE.extract_clusters_from_hierarchy(inp,
                                                        var_names, time_granularity, cluster_granularity)
############################################################################
############################################################################
############################################################################

####################################################
############# PLOTTING PATTERNS ####################
####################################################
# for i in range(0,8):
#     p = i
#     plt.figure()
#     plt.plot(inp[0,patterns[:,1]==p+1].T,'y')
#     plt.plot(patterns_dicts[1]['means'][p],'r')
#     plt.plot(patterns_dicts[1]['means'][p] + 2*patterns_dicts[1]['std'][p],'k')
#     plt.plot(patterns_dicts[1]['means'][p] - 2*patterns_dicts[1]['std'][p],'k')
# plt.show()
####################################################
####################################################
####################################################


############################################################################
########################## GROWING TREE ####################################
############################################################################  
tree_input = np.append(patterns,np.array([targets]).T,axis=1)
tree = Tree.DecisionTree(tree_input,depth,patterns_names,'pattern') 
paths = tree.returnPaths(0.3, 0.05)
############################################################################
############################################################################
############################################################################


############################################################################
####################### VISUALIZING RESULTS ################################
############################################################################
fig, axis = plt.subplots(3,4)
axis = axis.ravel()
var_ylims = []
for i in range(0,inp.shape[0]):
    axis[i].set_title(var_names[i] + ' CS')
    axis[i].plot(inp[i,targets==False].T,'y')
    axis[i].plot(inp[i,targets].T,'r')
    var_ylims.append([np.min(inp[i]),np.max(inp[i])])


for path in paths:
    number_nodes = len(path)
    fig, axis = plt.subplots(2,int((number_nodes+1)/2))
    axis = axis.ravel()
    i = 0
    for node in reversed(path):
        ftr_index = node['feature']
        ptn_index = node['pattern'] - 1
        var_index = patterns_var[ftr_index][0]
        T_int = patterns_var[ftr_index][1]
        x_axis = range(T_int[0],T_int[1])
        axis[i].set_xlim(0,50)
        #axis[i].set_ylim(var_ylims[var_index][0],var_ylims[var_index][1])
        mean = patterns_dicts[ftr_index]['means'][ptn_index]
        std = patterns_dicts[ftr_index]['std'][ptn_index]
        if node['in']:
            title_text = 'IN ' + var_names[var_index] + ': Score: ' + str(int(node['score'])) + ': Pur: ' + str(round(node['purity'],2)) + ' Cov: ' + str(round(node['coverage'],2))
            axis[i].plot(x_axis,mean,'r')
            axis[i].plot(x_axis,(mean + std),'k')
            axis[i].plot(x_axis,(mean - std),'k')
        else:
            title_text = 'OUT ' + var_names[var_index] + ': Score: ' + str(int(node['score'])) + ': Pur: ' + str(round(node['purity'],2)) + ' Cov: ' + str(round(node['coverage'],2))
            axis[i].plot(x_axis,mean,'y')
            axis[i].plot(x_axis,(mean + std),'y')
            axis[i].plot(x_axis,(mean - std),'y')
            
        axis[i].set_title(title_text + ' CS')
        #axis[i].set_yscale('log')
        
        i+=1
         
plt.show()
############################################################################
############################################################################
############################################################################   


