import gc
import traceback
from queue import Queue
from threading import Thread

import torch
import transformers

import modules.shared as shared


class _StopEverythingStoppingCriteria(transformers.StoppingCriteria):
    def __init__(self):
        # transformers.StoppingCriteria.__init__(self)
        super().__init__() # 不需要显式传递类名和实例对象。

    def __call__(self, input_ids: torch.LongTensor, _scores: torch.FloatTensor) -> bool:
        return shared.stop_everything


class Stream(transformers.StoppingCriteria):
    """
    Streams the output of a model to a callback function.
    StoppingCriteria可用于更改何时停止生成(EOS令牌除外)。
    """
    def __init__(self, callback_func=None):
        self.callback_func = callback_func

    def __call__(self, input_ids, scores) -> bool:
        if self.callback_func is not None:
            self.callback_func(input_ids[0])
        return False


class Iteratorize:

    """
    Transforms a function that takes a callback
    into a lazy iterator (generator).

    Adapted from: https://stackoverflow.com/a/9969000

    Iteratorize 类的作用是将一个接受回调函数参数的函数转换为惰性迭代器（generator）。
    Iteratorize 类可以将一个需要回调函数作为参数的函数进行转换。通过使用 Iteratorize 类，
    我们可以以惰性迭代器的形式来调用原始函数，并逐步获取结果，而不是一次性获取全部结果。
    对于处理大量数据或长时间运行的操作非常有用，可以减少内存消耗并提供灵活的结果获取方式。

    在给定的代码中，Iteratorize 类的初始化方法 __init__() 接收几个参数：
    func（原始函数）、args（位置参数）、kwargs（关键字参数）和 callback（回调函数）。
    在初始化过程中，它会创建一个队列（self.q）和一个线程（self.thread），并启动线程来执行原始函数。

    在 gentask() 方法中，原始函数被调用，并传递了一个名为 _callback 的回调函数作为参数。
    这个回调函数将接收到的值放入队列中。当发生特定条件时（self.stop_now 或 shared.stop_everything 标志为真），
    会抛出 ValueError 异常，从而终止迭代过程。

    通过实现 __iter__() 和 __next__() 方法，Iteratorize 类成为一个可迭代对象，可以通过迭代操作来获取函数执行的结果。
    每次调用 __next__() 方法时，会从队列中获取一个值，直到遇到 self.sentinel 标志，表示迭代结束。

    此外，Iteratorize 类实现了 __del__() 方法，在对象被销毁时执行清理操作，
    调用 clear_torch_cache() 函数来清理 Torch 和 CUDA 的缓存。

    当一个对象实现了 __enter__() 和 __exit__() 方法时，可以使用 with 语句来创建一个上下文环境，
    并在进入和离开该环境时执行相应的操作。通过将实例放置在 with 语句块中，可以确保在离开该块时会自动
    触发 __exit__() 方法
    with Iteratorize(func, args, kwargs, callback) as iterator:
      在此处进行迭代操作

    """

    def __init__(self, func, args=None, kwargs=None, callback=None):
        self.mfunc = func
        self.c_callback = callback
        self.q = Queue()
        self.sentinel = object()
        self.args = args or []
        self.kwargs = kwargs or {}
        self.stop_now = False

        def _callback(val):
            if self.stop_now or shared.stop_everything:
                raise ValueError
            self.q.put(val)

        def gentask():
            try:
                ret = self.mfunc(callback=_callback, *args, **self.kwargs)
            except ValueError:
                pass
            except:
                traceback.print_exc()
                pass

            clear_torch_cache()
            self.q.put(self.sentinel)
            if self.c_callback:
                self.c_callback(ret)
                
        # 开启线程，避免阻塞主线程
        self.thread = Thread(target=gentask)
        self.thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        obj = self.q.get(True, None)
        if obj is self.sentinel:
            raise StopIteration
        else:
            return obj

    def __del__(self):
        clear_torch_cache()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_now = True
        clear_torch_cache()


def clear_torch_cache():
    gc.collect()
    if not shared.args.cpu:
        torch.cuda.empty_cache()
