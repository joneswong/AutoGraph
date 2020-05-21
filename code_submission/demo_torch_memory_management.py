import numpy as np
import torch
from pynvml import *
nvmlInit()

from algorithms.gcn_algo import GCN

device = torch.device('cuda:0')

print("*********** 0 ***********")
#print(torch.cuda.get_device_properties(0).total_memory)
#print(torch.cuda.memory_cached(0))
print(torch.cuda.memory_cached(0))
#print(torch.cuda.max_memory_cached(0))
print(torch.cuda.memory_allocated(0))
#print(torch.cuda.max_memory_allocated(0))
print("**********************")
h = nvmlDeviceGetHandleByIndex(0)
info = nvmlDeviceGetMemoryInfo(h)
print(f'total    : {info.total}')
print(f'free     : {info.free}')
print(f'used     : {info.used}')
print("*********** 0 ***********")

ts = list()
for i in range(100):
    a_cpu = torch.ones((26843540,), dtype=torch.float32)
    a_gpu = a_cpu.to(device)
    ts.append(a_gpu)
    print("********** {} ************".format(i))
    #print(torch.cuda.get_device_properties(0).total_memory)
    #print(torch.cuda.memory_cached(0))
    print(torch.cuda.memory_cached(0))
    print(torch.cuda.max_memory_cached(0))
    print(torch.cuda.memory_allocated(0))
    print(torch.cuda.max_memory_allocated(0))
    print("**********************")
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    print(f'total    : {info.total}')
    print(f'free     : {info.free}')
    print(f'used     : {info.used}')
    print("*********** {} ***********".format(i))
    if i % 10 == 0:
        input()
del ts[:50]
print("********** del half before empty ************")
#print(torch.cuda.get_device_properties(0).total_memory)
#print(torch.cuda.memory_cached(0))
print(torch.cuda.memory_cached(0))
print(torch.cuda.max_memory_cached(0))
print(torch.cuda.memory_allocated(0))
print(torch.cuda.max_memory_allocated(0))
print("**********************")
h = nvmlDeviceGetHandleByIndex(0)
info = nvmlDeviceGetMemoryInfo(h)
print(f'total    : {info.total}')
print(f'free     : {info.free}')
print(f'used     : {info.used}')
print("*********** del half before empty ***********")
input()

"""
torch.cuda.empty_cache()
print("********** del half after empty ************")
#print(torch.cuda.get_device_properties(0).total_memory)
#print(torch.cuda.memory_cached(0))
print(torch.cuda.memory_cached(0))
print(torch.cuda.max_memory_cached(0))
print(torch.cuda.memory_allocated(0))
print(torch.cuda.max_memory_allocated(0))
print("**********************")
h = nvmlDeviceGetHandleByIndex(0)
info = nvmlDeviceGetMemoryInfo(h)
print(f'total    : {info.total}')
print(f'free     : {info.free}')
print(f'used     : {info.used}')
print("*********** del half after empty ***********")
input()
"""

estimate_available = info.free + (torch.cuda.memory_cached(0) - torch.cuda.memory_allocated(0))
print(estimate_available)
a_cpu = torch.ones((int((0.9*estimate_available)/4),), dtype=torch.float32)
a_gpu = a_cpu.to(device)

print("********** success or fail ************")
#print(torch.cuda.get_device_properties(0).total_memory)
#print(torch.cuda.memory_cached(0))
print(torch.cuda.memory_cached(0))
print(torch.cuda.max_memory_cached(0))
print(torch.cuda.memory_allocated(0))
print(torch.cuda.max_memory_allocated(0))
print("**********************")
h = nvmlDeviceGetHandleByIndex(0)
info = nvmlDeviceGetMemoryInfo(h)
print(f'total    : {info.total}')
print(f'free     : {info.free}')
print(f'used     : {info.used}')
print("*********** success or fail ***********")
input()


"""
a = GCN(2, 123)
b = list(a.parameters())
num_floats = 0
for param in b:
    print(param.size())
    tmp = np.prod(param.size())
    num_floats += tmp
    print(tmp)
print(num_floats)

mye = GCN.estimate_mem_consumption(10000, 600000, 2, 123)
print(torch.cuda.get_device_properties(0).total_memory)
print(torch.cuda.memory_cached(0))
print(torch.cuda.memory_allocated(0))
print(mye)
"""
