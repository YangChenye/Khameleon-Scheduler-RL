# Created by Chenye Yang on 2020/8/22.
# Copyright © 2020 Chenye Yang. All rights reserved.

from multiprocessing import Pool, cpu_count

print('num {}'.format(cpu_count()))

def fun(x):
    return x*x

with Pool(processes=8) as pool:
    a = pool.map(fun, range(100))