'''
Created on 23 Jan 2018

@author: Janis
'''

import numpy as np
import sobol

def compute_cluster_stats(data,cluster):
    unique_elements = np.unique(cluster)
    U, T = unique_elements.shape[0], data.shape[1]
    
    cluster_stats_dict = {
        'unique_elements': unique_elements,
        'means': np.zeros([U,T]),
        'std': np.zeros([U,T])
        }
    
    for i in range(0,U):
        unique = cluster_stats_dict['unique_elements'][i]
        mask = cluster == unique
        cluster_stats_dict['means'][i] = np.mean(data[mask],axis=0)
        cluster_stats_dict['std'][i] = np.std(data[mask],axis=0)
        
    return cluster_stats_dict

def get_sobol_intervals(T_steps, time_granularity):
    
    intervals = []
    for i in range(0,time_granularity):
            positions = sobol.generate_array(1, i,skip=1)
            positions = np.append(positions,[1])
            positions = np.sort(positions,axis=0)
            for j in range(0,positions.shape[0]):
                if j == 0:
                    last_index = 0
                else:
                    last_index = int(positions[j-1]*T_steps)
                current_index = int(positions[j]*T_steps) - 1
                new_interval = [last_index,current_index]
                if not any((new_interval == x) for x in intervals):
                    intervals.append([last_index,current_index])
                    
    return intervals
                
                
