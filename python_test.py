'''
@Author: lh-13
@Date: 2020-10-15 10:18:00
@LastEditTime: 2020-10-15 10:43:40
@LastEditors: Please set LastEditors
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
str1 = "key1 | key2 | key3"
str2 = "value1 | value2 | value3"

keys = str1.split('|')
values = str2.split('|')

d = dict(zip(keys, values))

for k, v in d.items():
    print(k, v)





