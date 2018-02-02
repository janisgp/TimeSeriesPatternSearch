'''
Created on 15 Jan 2018

@author: Janis
'''

import pandas as pd
import numpy as np

def identify_targets(results, interval_of_interest=[0,10]):
    # Computes the (binary) target values; 1 if in intervall_of_interest, 0 else
    # input:     results -> panel with 1-d results entries
    #            interval_of_interest -> list of two percentage values
    #                                    first one: lower threshold of percentage result values
    #                                    second one: upper threshold of percentage result values
    # output:    targets -> panel containing: 1 if value in intervall_of_interest
    #                                         0 else
    
    lower, upper = np.percentile(results,interval_of_interest)
    targets = np.multiply(np.array(lower<=results),np.array(upper>=results))
    return targets

def load_data_to_numpy(path_data, path_results, result_col_name='pvfp_at',result_oneIndex=False, dec_sep=';'):

    if result_oneIndex:
        results = pd.read_csv(path_results, index_col=[0], sep=dec_sep, thousands=',')
    else:
        results = pd.read_csv(path_results, index_col=[0,1], sep=dec_sep, thousands=',')
    scenarios = pd.read_csv(path_data, index_col=[0,1], sep=';')
    
    variables = scenarios.columns
    
    V, N, T = len(variables), scenarios[variables[0]].index.levels[0].size, scenarios[variables[0]].index.levels[1].size

    R = results[result_col_name]
    if result_oneIndex:
        output = R.values
    else:
        output = R.values.reshape([N,T])[:,0]
    print(output.shape)
    data = np.zeros([V, N, T])
    
    for i in range(0,V):
        data[i] = scenarios[variables[i]].values.reshape([N,T])
    
    return data, output, variables
    

def load_data(path_data, path_results, target_lvl):
    
    results = pd.read_csv(path_results, index_col=[0,1], sep=";", thousands=',')
    scenarios = pd.read_csv(path_data, index_col=[0,1], sep=";")

    N = int(scenarios.loc[:].shape[0] / scenarios.loc[1].shape[0])
    T = scenarios.loc[1].shape[0]
      
    index = [range(1,N+1),range(1,T+1)]
    pd_index = pd.MultiIndex.from_product(index, names=['first', 'second'])
    results_indexed = pd.DataFrame(results,pd_index)
    results_pvfp = results.loc[(slice(None),slice(0)),['pvfp_at']]/1e+6
      
    results_np = results_pvfp.as_matrix()
    min_result = results_np.min()
    target = results_np <= target_lvl*min_result
    
    N = target.shape[0]
    
    print('Amount of targets with low pvfp: ' + str(np.sum(target)))
    
    data_ir = scenarios['IR'].as_matrix()
    data_ir = np.reshape(data_ir,(N, int(data_ir.shape[0]/N)))
    data_eq = scenarios['EQ'].as_matrix()
    data_eq = np.reshape(data_eq,(N, int(data_eq.shape[0]/N)))
    
    data = dict(IR= data_ir,EQ= data_eq,)
    
    return data, target
    
    