import numpy as np
import LoadData as LD
from matplotlib import pyplot as plt
import dtw

def running_mean_var(data,window):
    V,N,T = data.shape
    run_mean = np.zeros([V,N,T-window+1])
    for i in range(0,V):
        run_mean[i] = running_mean(data[i],window)
    return run_mean

def running_mean (values, window):
    weights = np.repeat(1.0, window)/window
    N,T = values.shape
    run_mean = np.zeros([N,T-window+1])
    for i in range(0,N):
        run_mean[i] = np.convolve(values[i], weights, 'valid')
    return run_mean

scenarios_path = "OW-2017Q4_Scenarios_withoutCS.csv"
results_path = "OW-2017Q4_Results.csv"
inp, out, var_names = LD.load_data_to_numpy(scenarios_path, results_path,
                                            result_col_name="PVSP",result_oneIndex=True,dec_sep=';')

targets = LD.identify_targets(out,[0,10])
print(targets)
i1 = 2
i2 = 27
a = dtw.fastdtw(inp[0,i1,0:25],inp[0,i2,0:25],'cosine')

print(a[0])

plt.figure()
plt.plot(inp[0,i1,0:25])
plt.plot(inp[0,i2,0:25])
plt.show()