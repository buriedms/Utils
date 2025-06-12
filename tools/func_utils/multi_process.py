"""
多线程/多进程装饰器与函数工具（推荐装饰器用法）

功能：
- 提供@multi_process装饰器，可快速将任意函数并发执行。
- 也可直接调用multi_process_func(func, data_list, total_thread=3)。

用法示例：
    from func_utils.multi_process import multi_process

    @multi_process(total_thread=4)
    def my_func(sub_data):
        ...
    result = my_func(data_list)

依赖：threading, numpy, time
"""
import threading
import numpy as np
from time import sleep, ctime
from typing import Callable, List, Any, Optional
import pytest

# 通用性增强说明：
# 1. 支持任何可迭代数据（list/tuple/np.ndarray等），并自动分组。
# 2. 支持线程数自动调整（不超过数据长度）。
# 3. 支持函数带额外参数（*args, **kwargs），装饰器和函数接口均可用。
# 4. 线程安全收集结果，顺序不保证（如需顺序可后处理）。
# 5. 适合cli_tools等所有批量处理场景。

def multi_process_func(func: Callable, data_list, total_thread: int = 3, *args, **kwargs):
    """
    通用多线程主控函数
    :param func: 任务函数，接收一个子列表
    :param data_list: 可迭代数据（list/tuple/np.ndarray等）
    :param total_thread: 线程数
    :param args: 传递给func的额外参数
    :param kwargs: 传递给func的额外参数
    :return: 所有线程的输出结果列表
    """
    data_list = list(data_list)
    lenList = len(data_list)
    total_thread = min(total_thread, lenList) if lenList > 0 else 1
    gap = int(np.ceil(lenList / total_thread))
    threads = []
    out_list = []
    class MyThread(threading.Thread):
        def __init__(self, threadID, name, sub_data, out_list):
            super().__init__()
            self.threadID = threadID
            self.name = name
            self.sub_data = sub_data
            self.threadLock = threading.Lock()
            self.out_list = out_list
        def run(self):
            self.threadLock.acquire()
            self.out_list.append(func(self.sub_data, *args, **kwargs))
            self.threadLock.release()
    for i in range(total_thread):
        sub_data = data_list[i*gap : min((i+1)*gap, lenList)]
        thread = MyThread(i, f"Thread-{i}", sub_data, out_list)
        threads.append(thread)
    for thread in threads:
        thread.start()
    for t in threads:
        t.join()
    return out_list

def multi_process(total_thread: int = 3):
    """
    通用多线程装饰器，自动将函数并发执行
    :param total_thread: 线程数
    用法：@multi_process(total_thread=4)
    """
    def decorator(func):
        def wrapper(data_list, *args, **kwargs):
            return multi_process_func(lambda sub_data, *a, **k: func(sub_data, *a, **k), data_list, total_thread=total_thread, *args, **kwargs)
        return wrapper
    return decorator


def test_multi_process_decorator():
    @multi_process(total_thread=2)
    def func(sub_data):
        return [x + 1 for x in sub_data]
    data = [1, 2, 3, 4]
    result = func(data)
    # 检查结果类型和内容
    assert isinstance(result, list)
    assert sorted(sum(result, [])) == [2, 3, 4, 5]

def test_multi_process_func():
    def func(sub_data):
        return [x * 2 for x in sub_data]
    data = [1, 2, 3, 4]
    result = multi_process_func(func, data, total_thread=2)
    assert isinstance(result, list)
    assert sorted(sum(result, [])) == [2, 4, 6, 8]
