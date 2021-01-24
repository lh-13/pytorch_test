'''
@Author: lh-13
@Date: 2020-10-15 10:18:00
@,@LastEditTime: ,: 2020-11-02 23:03:49
@,@LastEditors: ,: Please set LastEditors
@Description: 测试python的一些方法与功能实现
@FilePath: \pytorch_test\python_test.py
'''

#拆分字符串入到字典里面

# str = "key1=value1;key2=value2;key3=value3"
# d = dict(x.split('=') for x in str.split(';'))
# for (k,v) in d.items():
#     print(k, v)

#输出量：
# key1 value1
# key2 value2
# key3 value3

#将两个列表转换成字典
# str1 = "key1 | key2 | key3"
# str2 = "value1 | value2 | value3"

# keys = str1.split('|')
# values = str2.split('|')

# d = {}
# for key in keys:
#     key = key.strip()  #key trim 
#     for v in values:
#         d[key] = v.strip()    #value trim 

# for k, v in d.items():
#     print(k, v)

#输出量：
# key1 value1
# key2 value2
# key3 value3

#zip 示例
# str1 = "key1 | key2 | key3"
# str2 = "value1 | value2 | value3"

# keys = str1.split('|')
# values = str2.split('|')

# d = dict(zip(keys, values))

# for k, v in d.items():
#     print(k, v)


#---------------------------------------------------面试题
#给个有序数组，然后求元素平方后不重复的元素个数，例如[-10, -10, -5, 0, 1, 5, 8, 10]
flag = 0 
if flag == 1:
    data = [-10, -10, -5, 0, 1, 5, 8, 10]
    print(len(set([x**2 for x in data])))
#https://www.runoob.com/python3/python3-set.html



#----------------------------------------------------------面试题append()、extend()逐个添加
#list.append(obj) 向列表中添加一个对象object,object做为一个整体（如果添加的是列表，则会带有[]）   整体添加
#list.extend(sequence) 把一个序列seq的内容添加到列表中    逐个添加
flag = 1
if flag == 1:
    #append()
    array = [1, 2, 3]
    array.append([4 , 5, 6])   #[1, 2, 3, [4, 5, 6]]
    #array.append(4)    #[1, 2, 3, 4]
    #array.append('4')   #[1, 2, 3, '4']
    print(array)

    #extend
    array_1 = [1, 2, 3]
    array_1.extend([4, 5 ,6])   #[1, 2, 3, 4, 5, 6]
    #array_1.extend([4])   #[1, 2, 3, 4]
    #array_1.extend('4')   #[1, 2, 3, '4']
    print(array_1)













