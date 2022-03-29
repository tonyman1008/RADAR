import numpy as np
import torch


h_list = [10,20,30]
total_h = sum(h_list)
component_size = len(h_list)

# for i in range(component_size):
#     h_start = sum(h_list[:i])
#     h_end = sum(h_list[:i+1])
#     print("h_start",h_start)
#     print("h_end",h_end)

for test in range(len(h_list)):
    h_list[test] += 10

print(h_list)
