'''
@,@Author: ,: lh-13
@,@Date: ,: 2020-11-03 13:46:49
@,@LastEditTime: ,: 2020-11-08 10:38:39
@,@LastEditors: ,: Please set LastEditors
@,@Description: ,: pytorch 实例100
@,@FilePath: ,: \pytorch_test\python_test_100_example.py
'''


#-------------------------------------------------------1.hello world 


#-------------------------------------------------------2. 数字求和
#通过用户输入两个数字，并计算两个数字之和
flag = 0
if flag == 1:
    num1 = input('请输入第一个数字：')
    num2 = input('请输入第二数字：')

    sum = float(num1) + float(num2)
    print('sum=%.2f'%sum)

flag = 0
if flag == 1:
    #将上面的代码简化为一行
    print('sum=%.2f'%(float(input('请输入第一个数字：'))+float(input('请输入第二个数字：'))))


#-------------------------------------------------------3.平方根
#通过用户输入一个数字，并计算这个数字的平方根
flag = 0       
if flag == 1:
    num = input('请输入一个数字：')
    #sqrt = float(num)**0.5     #** 只适用于正数。负数和复数会出错
    #print('其平方根=', sqrt)
    import cmath   #导入复数数学模块
    sqrt = cmath.sqrt(float(num))
    print('其平方根为，实数部分{},复数部分为{}'.format(sqrt.real, sqrt.imag))


#-------------------------------------------------------4. 二次方程
#通过用户输入数字，并计算二次方程：
#二次方程式 ax**2 + bx + c = 0   
#a、b、c 用户提供，为实数，a != 0 
#求根公式：1.(-b+sqrt(b**2-4ac))/2a
#          2.(-b-sqrt(b**2-4ac))/2a   
flag = 0  
if flag == 1: 
    import cmath  
    a = float(input('请输入a的值：'))
    b = float(input('请输入b的值：'))
    c = float(input('请输入c的值：'))
    #计算  
    d = (b**2)-4*a*c    
    
    sol1 = (-b+cmath.sqrt(d))/(2*a)    
    sol2 = (-b-cmath.sqrt(d))/(2*a)

    print('二元一次方程的解为{},{}'.format(sol1, sol2))


#-------------------------------------------------------5. 计算三角形的面积 
flag = 0 
if flag == 1:   
    a = float(input('请输入三角形的边长a:'))
    b = float(input('请输入三角形的边长b：'))
    c = float(input('请输入三角形的边长c:'))

    # h = cmath.sqrt(a**2-c**2)    这样是已知了三角形的高与底
    # area = h*c/2  
    #计算半周长
    s = (a+b+c)/2       
    #计算面积 
    area = (s*(s-a)*(s-b)*(s-c))**0.5        
    print('area:%.2f' % area)


#-------------------------------------------------------6. 计算圆的面积 
#定义一个函数实现求圆的面积

flag = 0  
def findArea(r):
    PI = 3.142
    return PI*r*r   

if flag == 1:
    print('圆的面积为：%.6f'%findArea(5))

#-------------------------------------------------------7. python 随机数生成
#生成0~9之间的随机数
flag = 0  
if flag == 1:
    import random           
    print(random.randint(0,9))

#-------------------------------------------------------8. 判断字符串是否为数字
flag = 0  
def is_number(s):
    try:
        float(s)   #如果这里不报错，则说明可以转换成数字，则为数字
        return True
    except ValueError:     #python 常用标准异常，表示传入的参数无效
        pass   #什么也不做

    try:
        import unicodedata  #处理ascii码的包
        unicodedata.numeric(s)
        return True  
    except (TypeError, ValueError):
        pass  
    
    return False    

if flag == 1:
    print(is_number('foo'))
    print(is_number(1))
    print(is_number(-1.37))
    print(is_number(1e3))    
    print(is_number('.'))


#-------------------------------------------------------9. 斐波那契数列
#斐波那契数列指的是这样一个数列 0, 1, 1, 2, 3, 5, 8, 13,特别指出：第0项是0，第1项是第一个1。从第三项开始，每一项都等于前两项之和
flag = 0 
def fibonacci(n):
    if n <= 1:
        return n   
    else:  
        return fibonacci(n-1)+fibonacci(n-2)

if flag == 1: 
    for i in range(0, 8):
        print(fibonacci(i), end=' ')

#-------------------------------------------------------10. python 约瑟夫生者死者小游戏
'''
30个人在一条船上，超载，需要15人下船。
于是人们排成一队，排队的位置即为他们的编号
报数，从1开始，数到9的人下船，接着重新开始从1报数，如此循环，直到船上仅剩15人为止，
问都 有哪些人下了船
'''
flag = 0
if flag == 1:
    num = list(range(1, 31, 1))
    for i in range(30):
        num[i] = 1   #创建一个有30个元素的列表，每个位置的值为1，代表人在船上，当值为0时则人已经下船了
    
    go_num = 0
    start_num = 0
    i = 0
    while(go_num < 15):   #可以报数
        checked = 0
        #for i in range(start_num, 30):
        while (i <= 30):
            if num[i] == 1:   #还在船上，报数有效
                checked += 1
            # else:
            #     continue
            if checked == 9:
                num[i] = 0
                print(i+1, '号下船了\n')     #因为索引为0-29
                start_num = i+1
                go_num += 1
                break   #此次报数结束，重新开始下一轮报数
            i+=1
            if i == 30:   #一轮报数结束，需要从头开始接着报
                start_num = -1
                i = 0

#-------------------------------------------------------11. python 数组翻转指定个数的元素
'''
定义一个整型数组，并将指定个数的元素翻转到数组的尾部。

例如：(ar[], d, n) 将长度为 n 的 数组 arr 的前面 d 个元素翻转到数组尾部。

以下演示了将数组的前面两个元素放到数组后面。

原始数组:
[1,2,3,4,5,6,7]
翻转后：
[3, 4, 5, 6, 7, 1, 2]
'''

'''
解题思路：
分别循环d次，每次将d中第一个元素先储存在temp中，然后d之后的元素依次往前挪移一个
'''


flag = 0 
def leftRotate(arr, d, n):
    for i in range(d):
        leftRotatebyone(arr, n)

def leftRotatebyone(arr, n):
    temp = arr[0]
    for i in range(n-1):
        arr[i] = arr[i+1]
    arr[n-1] = temp   

def printArr(arr, size):
    for i in range(size):
        print(arr[i])

if flag == 1:
    arr = [1, 2, 3, 4, 5, 6, 7]
    leftRotate(arr, 2, len(arr))
    printArr(arr, len(arr))

#-------------------------------------------------------11. python 将列表中头尾两个元素对调
'''
定义一个列表，交将头尾两个元素对调
例如：
[1, 2, 3]
对调后：
[3, 2, 1]
'''

flag = 0 

def SwapHeadTail(arr):
    head = arr[0]
    tail = arr[-1]
    arr[0] = tail
    arr[-1] = head  
    return arr   
if flag == 1:
    arr = [1, 2, 3]
    SwapHeadTail(arr)
    print(arr)
    

#-------------------------------------------------------12.python 将列表中指定位置的两个元素对调
'''
定义一个列表，并将列表中的指定位置的两个元素对调。

例如，对调第一个和第三个元素：

对调前 : List = [23, 65, 19, 90], pos1 = 1, pos2 = 3
对调后 : [19, 65, 23, 90]
'''

flag = 0  
def SwapList(arr, pos1, pos2):
    arr[pos1], arr[pos2] = arr[pos2], arr[pos1]
    return arr  
if flag == 1:
    arr = [23, 65, 19, 90]
    print(arr)
    SwapList(arr, 1, 3)
    print(arr)


#-------------------------------------------------------12.python 翻转列表
'''
定义一个列表，并将它翻转。

翻转前 : list = [10, 11, 12, 13, 14, 15]
翻转后 : [15, 14, 13, 12, 11, 10]
'''

flag = 0
def ReverseList(arr):
    return [ele for ele in reversed(arr)]

if flag == 1:
    arr = [10, 11, 12, 13, 14, 15]
    print(ReverseList(arr))

#-------------------------------------------------------13.python 实现二分法查找
'''
二分搜索是一种在有序数组中查找某一特定元素的搜索算法。搜索过程从数组的中间元素开始，
如果中间元素正好是要查找的元素，则搜索过程结束；如果某一特定元素大于或者小于中间元素，
则在数组大于或小于中间元素的那一半中查找，而且跟开始一样从中间元素开始比较。如果在某一步骤数组为空，
则代表找不到。这种搜索算法每一次比较都使搜索范围缩小一半。
详见：binary_serarch_method.jpg
'''
flag = 1 
def binary_serarch(arr, l, r, x):
    #l:最左边界 r:最右边界 x:为要查找的值
    if r >= l:
        mid = int(l+(r-l)/2)

        if arr[mid] == x:   #刚好在中间
            return mid 
        elif arr[mid] > x:  #x在左边，此时需要调整r的值
            return binary_serarch(arr, l, mid, x)
        elif arr[mid] < x:  #x在右边，此时需要调整l的值为mid
            return binary_serarch(arr, mid, r, x)     

    else:
        print('不存在')
        return -1 

if flag == 1:
    arr = [2, 3, 4, 10, 40]
    x = 10 
    result = binary_serarch(arr, 0, len(arr)-1, x)
    if result != -1:
        print('x所在的索引为：%d' % result)
    else:
        print('元素没有在数组中')



