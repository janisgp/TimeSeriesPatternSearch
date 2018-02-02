'''
Created on 15 Jan 2018

@author: Janis
'''

import numpy as np
from datashape.typesets import boolean

def get_stats_per_pattern(targets, patterns):
    # assumes targets 0 and 1; looks for targets of 1
    unique_patterns = np.unique(patterns)
    D = unique_patterns.shape[0]
    N = targets.shape[0]
    number_targets = np.sum(targets)
    unique_masks = np.zeros([D,N], dtype=bool)
    purity, coverage = np.zeros(D), np.zeros(D)
    for i in range(0,D):
        unique_masks[i] = (patterns == unique_patterns[i])
        target_count = np.sum(targets[unique_masks[i]])
        general_count = np.sum(unique_masks[i])
        purity[i] = target_count/general_count
        coverage[i] = target_count/number_targets
    return unique_patterns, purity, coverage, unique_masks
        