'''
@,@Author: ,: lh-13
@,@Date: ,: 2020-11-03 13:46:49
@,@LastEditTime: ,: 2020-11-04 17:26:11
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
flag = 1
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




