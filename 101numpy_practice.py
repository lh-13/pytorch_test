'''
@version        : 1.0
@Author         : lh-13
@Date           : 2021-02-07 13:39:13
@LastEditors    : lh-13
@LastEditTime   : 2021-02-07 13:39:13
@Descripttion   : 101道numpy 练习题
@FilePath       : \pytorch_test/101numpy_practice.py
'''

import numpy as np  

#1.import numpy and print the version number  
print(np.__version__)

#2.create a 1D array of numbers from 0 to 9
flag = 0
if flag == 1:
    arr = np.arange(10)
    print(arr)

#3.create a 3*3 numpy array of all True's
flag = 0  
if flag == 1:
    #ones_array = np.ones((3,3), dtype=np.bool)
    ones_array = np.full((3, 3), True, dtype=np.bool)
    print(ones_array)


#4. extract all odd numbers from arr 
flag = 1  
if flag == 1:
    arr = np.arange(10)
    #odd_array = arr[np.where(arr % 2 == 1)]
    odd_array = arr[arr % 2 == 1]
    print(odd_array)


