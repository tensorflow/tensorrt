#!/usr/bin/env python
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# -*- coding: utf-8 -*-

import time

import numpy as np
import tensorflow as tf

from benchmark_logger import logging
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
            start_t = time.perf_counter()
            output = self._fns[fn_id](*arg, **kwargs)
            self._timings[fn_id].append(time.perf_counter() - start_t)
        except IndexError:
            print()  # visual spacing
            logging.debug("AutoTuning is over... Collecting timing statistics:")
            perf_data = []
            for idx, fn_stat in enumerate(self._timings):
                perf_data.append(np.mean(fn_stat[self._skip_n_first:]))
                logging.debug(
                    f"\t- Function ID: {idx} - "
                    f"Name: {self._fns[idx].__name__:40s} - "
                    f"Average Exec Time: {perf_data[-1]}"
                )

            best_fn_id = np.argmin(perf_data)
            logging.debug(
                f"Selecting function ID: {best_fn_id}. "
                f"Setting exec path to: `{self._fns[best_fn_id].__name__}`\n"
            )

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
            logging.info(f"Building the concrete function")
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

        func_name = func.__name__

        eager_function = func

        tf_function = tf.function(jit_compile=use_xla)(func)

        def resync_gpu_wrap_fn(_func, str_appended):
            name = f"{func_name}_{str_appended}"
            _func.__name__ = name
            _func = force_gpu_resync(_func)
            _func.__name__ = name
            return _func

        eager_function = resync_gpu_wrap_fn(eager_function, "eager")
        tf_function = resync_gpu_wrap_fn(tf_function, "tf_function")

        funcs2autotune = [eager_function, tf_function]

        if use_synthetic_data:
            logging.debug(
                "Allowing direct concrete_function call with "
                "synthetic data loader."
            )

            tf_concrete_function = _force_using_concrete_function(
                tf.function(jit_compile=use_xla)(func)
            )
            tf_concrete_function = resync_gpu_wrap_fn(
                tf_concrete_function, "tf_concrete_function"
            )

            funcs2autotune.append(tf_concrete_function)

        return _TFFunctionAutoTuner(
            funcs2autotune,
            calls_per_func=calls_per_func,
            skip_n_first=skip_n_first
        )

    return wrapper
