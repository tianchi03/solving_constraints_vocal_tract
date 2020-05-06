import sys
import os

mypath = 'simulations_test/' #04_equations/'
f = []
for (dirpath, dirnames, filenames) in os.walk(mypath):
    f.extend(filenames)
    break

print(f)


#####
list_str = ['vocaltract_bla_N=12', 'vocaltract_bla_N=2',
            'vocaltract_bla_N=123']
N_list = []
for fn in list_str:
    for i, ch in enumerate(fn):
        if ch == '=':
            N_list.append(fn[i+1:])
print(N_list)

print('12' in N_list)
