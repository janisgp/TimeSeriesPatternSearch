import numpy as np

def identify_categorical_variables(data):
    # identifies whether a column contains categorical data
    # this is identified by checking whether the column purely
    # contains integers
    # input:     data -> pandas.panel
    # output:    dictionary (cat_dict) with following entries:
    #                if not categorical data -> None 
    #                else -> list with two numpy arrays (one containing
    #                        the unique categories, one containing
    #                        the corresponding number of occurrences
    
    cat_dict = {}
    data_dtypes = data.dtypes
    _, N, D = data.shape
    
    for column in data:
        if 'int' in data_dtypes[column].name:
            unique_elements, counts = np.unique(data[column].as_matrix(),return_counts=True)
            relative_counts = counts / (N*D)
            cat_dict[column] = [unique_elements,relative_counts]
        else:
            cat_dict[column] = None
            
    return cat_dict

def identify_targets(results, interval_of_interest=[0,0.1]):
    # Computes the (binary) target values; 1 if in intervall_of_interest, 0 else
    # input:     results -> panel with 1-d results entries
    #            interval_of_interest -> list of two decimal values
    #                                    first one: lower threshold of percentage result values
    #                                    second one: upper threshold of percentage result values
    # output:    targets -> panel containing: 1 if value in intervall_of_interest
    #                                         0 else
    
    lower, upper = results.quantile(q=interval_of_interest)
    targets = np.multiply(np.array(lower<=results),np.array(upper>=results))
    return targets
    
