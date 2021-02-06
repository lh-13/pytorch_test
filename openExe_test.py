'''
@version        : 1.0
@Author         : lh-13
@Date           : 2021-02-02 15:53:12
@LastEditors    : lh-13
@LastEditTime   : 2021-02-02 15:53:12
@Descripttion   : 捕获exe程序输出到dos界面的内容
@FilePath       : \pytorch_test\openExe_test.py
'''

import sys  
from subprocess import *  
proc = Popen('OpencvDnnYolov4.exe', bufsize=1024, stdin=PIPE, stdout=PIPE)
(fin, fout) = (proc.stdin, proc.stdout)
for i in range(100):
    print(fout.readline())
