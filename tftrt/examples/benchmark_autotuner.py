#!/usr/bin/env python
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# -*- coding: utf-8 -*-

import time
import numpy as np
import tensorflow as tf

from benchmark_utils import force_gpu_resync


class _TFFunctionAutoTuner(object):
    def __init__(self, funcs, calls_per_func, skip_n_first):
        if not isinstance(funcs, (tuple, list)):
            raise ValueError("Argument `funcs` must be a list or tuple.")

        if any([not callable(fn) for fn in funcs]):
            raise ValueError("One of the function passed is not callable.")

        self._fns = funcs
        self._calls_per_func = calls_per_func
        self._skip_n_first = skip_n_first

        self._call_counter = 0
        self._timings = [[] for _ in range(len(self._fns))]

        self._best_fn = self._autotune

    def _autotune(self, *arg, **kwargs):
        fn_id = self._call_counter // self._calls_per_func
        try:
            start_t = time.time()
            output = self._fns[fn_id](*arg, **kwargs)
            self._timings[fn_id].append(time.time() - start_t)
        except IndexError:
            print("\n[DEBUG] AutoTuning is over... Collecting timing statistics:")
            perf_data = []
            for idx, fn_stat in enumerate(self._timings):
                perf_data.append(np.mean(fn_stat[self._skip_n_first:]))
                print(f"\t- [DEBUG] Function ID: {idx} - "
                      f"Name: {self._fns[idx].__name__:40s} - "
                      f"Average Exec Time: {perf_data[-1]}")

            best_fn_id = np.argmin(perf_data)
            print(f"[DEBUG] Selecting function ID: {best_fn_id}. "
                  f"Setting exec path to: `{self._fns[best_fn_id].__name__}`\n")

            self._best_fn = self._fns[best_fn_id]
            return self._best_fn(*arg, **kwargs)

        self._call_counter += 1
        return output

    def __call__(self, *arg, **kwargs):
        return self._best_fn(*arg, **kwargs)


def _force_using_concrete_function(func):
    # `context` needs to be a closure of type list or dict for persistance
    context = []
    def _wrapper(*args, **kwargs):
        try:
            return context[0](*args, **kwargs)
        except IndexError:
            print(f"[INFO] Building the concrete function")
            context.append(func.get_concrete_function(*args, **kwargs))
            return context[0](*args, **kwargs)
    return _wrapper


def auto_tf_func_tuner(
    calls_per_func=45,
    skip_n_first=30,
    use_xla=False,
    use_synthetic_data=False
):

    def wrapper(func):

        @force_gpu_resync
        def eager_function(*args, **kwargs):
            return func(*args, **kwargs)

        @force_gpu_resync
        @tf.function(jit_compile=use_xla)
        def tf_function(*args, **kwargs):
            return func(*args, **kwargs)

        @force_gpu_resync
        @_force_using_concrete_function
        @tf.function(jit_compile=use_xla)
        def tf_concrete_function(*args, **kwargs):
            return func(*args, **kwargs)

        eager_function.__name__ = f"{func.__name__}_eager"
        tf_function.__name__ = f"{func.__name__}_tf_function"
        tf_concrete_function.__name__ = f"{func.__name__}_tf_concrete_function"

        funcs2autotune = [eager_function, tf_function]
        if use_synthetic_data:
            print("[INFO] Allowing direct concrete_function call with "
                  "synthetic data loader.")
            funcs2autotune.append(tf_concrete_function)

        return _TFFunctionAutoTuner(
            funcs2autotune,
            calls_per_func=calls_per_func,
            skip_n_first=skip_n_first
        )

    return wrapper
