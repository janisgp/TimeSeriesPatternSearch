'''
Created on 10 Jan 2018

@author: Janis
'''

import numpy as np
import sobol
#from fastdtw import fastdtw
from scipy.cluster.hierarchy import linkage, fcluster
import Tools

#  patterns.append([current_cluster_dict['cluster']])
#             patterns_dicts.append(current_cluster_dict)

def extract_clusters_from_hierarchy(data, var_names, time_granularity=2, cluster_granularity=0.1):
    
    V, _, T = data.shape
    T_intervals = Tools.get_sobol_intervals(T, time_granularity)
    
    print(T_intervals)
    
    patterns = []
    patterns_dicts = []
    patterns_var = []
    patterns_names = []
    
    for i in range(0,V):
        print(i)
        for T_int in T_intervals:
            current_cluster_dict = cluster_hierarchy(data[i,:,T_int[0]:T_int[1]], cluster_granularity)
            patterns.append(current_cluster_dict['cluster'])
            patterns_dicts.append(current_cluster_dict)
            patterns_names.append(var_names[i] + ': Pattern between t0 = ' + str(T_int[0]) + ' and t1 = ' + str(T_int[1]))
            patterns_var.append([i,T_int])
    
    return np.array(patterns).T, patterns_dicts, patterns_names, patterns_var

def cluster_hierarchy(data, cluster_granularity=0.1):
    
    N = data.shape[0]
    linkage_matrix = linkage(data, method='ward')
    cluster = fcluster(linkage_matrix, int(cluster_granularity*N), 'maxclust')
    cluster_dict = Tools.compute_cluster_stats(data, cluster)
    cluster_dict['cluster'] = cluster
    
    return cluster_dict


def extract_features(data, granularity, data_type):
    
    D = data.shape[1]
    
    #computing features
    mean = np.mean(data,axis=1)
    features = [mean] 
    features_names = [data_type + ': mean']
    std = np.std(data,axis=1)
    features = np.append(features,[std],axis=0)
    features_names.append(data_type + ': std')
    
    for i in range(0,granularity):
    
        positions = sobol.generate_array(1, 2**i,skip=1)
        positions = np.append(positions,[1])
        positions = np.sort(positions,axis=0)
    
        j = 0
        for pos in positions:
            
            if j == 0:
                last_index = 0
            else:
                last_index = int(positions[j-1]*D)
            
            current_index = int(pos*D) - 1
                
            mean = np.mean(data[:,last_index:current_index],axis=1)
            features = np.append(features,[mean],axis=0)
            features_names.append(data_type + ': mean inbetween year ' + str(last_index+1) + ' and ' + str(current_index+1))
            
            std = np.std(data[:,last_index:current_index],axis=1)
            features = np.append(features,[std],axis=0)
            features_names.append(data_type + ': std inbetween year ' + str(last_index+1) + ' and ' + str(current_index+1))
            
            rise = data[:,current_index] - data[:,last_index]
            features = np.append(features,[rise],axis=0)
            features_names.append(data_type + ': rise from year ' + str(last_index+1) + ' to ' + str(current_index+1))
            
            if current_index < D-1:
                slope = (data[:,current_index+1] - data[:,current_index-1])/2
            else:
                slope = (data[:,current_index] - data[:,current_index-1])
            features = np.append(features,[slope],axis=0)
            features_names.append(data_type + ': derivative at year ' + str(current_index+1))
            
            j+=1
    
    return features, features_names
'''
def extract_patterns(data, similarity_threshold, granularity, variable):
    
    N, D = data.shape
    patterns = np.array([np.zeros(N)])
    patterns_names = [variable + ': overall patterns']
    pattern_scores = np.ones(N)
    
    for i in range(0,N):
          
        print(i)
          
        time_series = data[i,:]
          
        for j in range(0,N):
              
            if pattern_scores[j] > similarity_threshold:
                score,_ = fastdtw(time_series,data[j,:])
                if score <= similarity_threshold:
                    patterns[0,j] = i+1
                    pattern_scores[j] = score
                    
    for i in range(0,granularity):
        positions = sobol.generate_array(1, 2**i,skip=1)
        positions = np.append(positions,[1])
        positions = np.sort(positions,axis=0)
    
        j = 0
        k = 0
        for pos in positions:
            
            if j == 0:
                last_index = 0
            else:
                last_index = int(positions[j-1]*D)
            
            current_index = int(pos*D) - 1
            
            patterns_names.append(variable + ': pattern between time ' + str(last_index) + ' and time ' + str(current_index))
            
            pattern_scores = np.ones(N)
            
            patterns = np.append(patterns, [np.zeros(N)], axis=0)
            
            dim = patterns.shape[0]
            
            for l in range(0,N):
                
                time_series = data[l,last_index:current_index]
                
                print(l)
                
                for k in range(0,N):
                
                    if pattern_scores[k] > similarity_threshold:
                        score,_ = fastdtw(time_series,data[k,last_index:current_index])
                        if score <= similarity_threshold:
                            patterns[dim-1,k] = l+1
                            pattern_scores[k] = score
            
            j+=1
        
    
    return patterns, patterns_names

def extract_relevant_patterns(data, target_mask, similarity_threshold, granularity, variable):
    
    N, D = data.shape
    patterns = np.array([np.zeros(N)])
    patterns_names = [variable + ': overall patterns']
    pattern_scores = np.ones(N)
    
    for i in range(0,N):
          
        print(i)
          
        time_series = data[i,:]
          
        for j in range(0,N):
              
            if pattern_scores[j] > similarity_threshold:
                score,_ = fastdtw(time_series,data[j,:])
                if score <= similarity_threshold:
                    patterns[0,j] = i+1
                    pattern_scores[j] = score
                    
    for i in range(0,granularity):
        positions = sobol.generate_array(1, 2**i,skip=1)
        positions = np.append(positions,[1])
        positions = np.sort(positions,axis=0)
    
        j = 0
        k = 0
        for pos in positions:
            
            if j == 0:
                last_index = 0
            else:
                last_index = int(positions[j-1]*D)
            
            current_index = int(pos*D) - 1
            
            patterns_names.append(variable + ': pattern between time ' + str(last_index) + ' and time ' + str(current_index))
            
            pattern_scores = np.ones(N)
            
            patterns = np.append(patterns, [np.zeros(N)], axis=0)
            
            dim = patterns.shape[0]
            
            for l in range(0,N):
                
                time_series = data[l,last_index:current_index]
                
                print(l)
                
                for k in range(0,N):
                
                    if pattern_scores[k] > similarity_threshold:
                        score,_ = fastdtw(time_series,data[k,last_index:current_index])
                        if score <= similarity_threshold:
                            patterns[dim-1,k] = l+1
                            pattern_scores[k] = score
            
            j+=1
        
    
    return patterns, patterns_names '''
    
    
    
    