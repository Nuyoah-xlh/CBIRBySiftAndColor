import time
from threading import Thread

import numpy as np
import pandas

global a
a=[]
def thread_function(age):
    global a
    if(age==0):
        time.sleep(2)
        a.append(111)
    else:
        a.append(2222)
def run_threading(target, args, count):
    """
    :param target: 目标函数
    :param args: 函数参数
    :param count: 线程数量
    """
    ts = []
    k=0
    for i in range(count):
        t = Thread(target=target, args=(k,))
        k+=1
        ts.append(t)
    [i.start() for i in ts]
    [i.join() for i in ts]

if __name__ == '__main__':
    ages = [1, 3, 4]
    # 1111
    run_threading(thread_function, (ages,), 2)
    print(a)
    # print(q)