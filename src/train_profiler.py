"""Used to construct the required profiler for target platform.
"""
from NFGen.data_preprocess import construct_km_profiler
from NFGen.profiler import SubPoly
import numpy as np
import pickle

data_folder = "./data/dis_profiler/"
profiler_folder = "./NFGen/PerformanceModel/"

# profiler for SPDZ.
file_list = ['repring', 'repprime', 'psrepring', 'psrepprime', 'shamir']
name_list = ['Rep3', 'RepPrime', 'PsRepRing', 'PsRepPrime', 'Shamir3']

k_list = [i for i in range(3, 10, 2)]
m_list = [i for i in range(2, 50, 2)]
repeat_list = [50, 10, 5, 5, 10]

for i in range(len(file_list)):
    f_name = file_list[i]
    name = name_list[i]
    repeat = repeat_list[i]
    
    data_file = data_folder + f_name + ".txt"
    construct_km_profiler(data_file, profiler_folder+name+"_kmProfiler.pkl", k_list, m_list, system="MP-SPDZ", repeat=repeat, degree=2)


# profiler for PrivPy.
k_list = [i for i in range(3, 10, 2)]
m_list = [i for i in range(2, 50, 5)]

data_list = [0.03055278062765865,0.036243653299607104,0.0578379988692177,0.0623798561091462,0.06792744159974973,0.07159075974959705,0.07507551431990578,0.07930161952936032,0.08473131656955957,0.08931634425971424,0.03065495967894094,0.037805907729307364,0.05924086570939835,0.0623550772697854,0.07017467498735641,0.0741860938096579,0.07767512320970127,0.08093848466887721,0.08825953006999043,0.08951034545953007,0.03241130113929103,0.03697632073999557,0.06010249137852952,0.06474335431994405,0.0703509378427043,0.07492305516916531,0.07752626180990774,0.0831897950192797,0.0881021928798873,0.09147178649982379,0.0336340713492973,0.0403621149052924,0.06250393866957893,0.06568181275997631,0.07114339827967342,0.07620105504975072,0.07971606254977814,0.08402011155976652,0.08897655486998701,0.09370103597939305]

x = []
for k in k_list:
    for m in m_list:
        x.append([m, k])
x = np.array(x)
y = np.array(data_list)

km_profiler = SubPoly(degree=3, fit_intercept=True)
km_profiler.fit(x, y)
# km_profiler.model_analysis(x, y)

p = open(profiler_folder + 'PrivPy_KMProfiler.pkl', 'wb')
pickle.dump(km_profiler, p)
print("----> Save model in ", profiler_folder + 'PrivPy_KMProfiler.pkl')