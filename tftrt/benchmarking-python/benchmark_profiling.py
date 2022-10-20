#!/usr/bin/env python
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# -*- coding: utf-8 -*-

import functools
import contextlib

import tensorflow as tf

from benchmark_logger import logging
from benchmark_utils import timed_section


class ProfilingCTX(object):

    def __init__(self, export_dir=None, verbose=False, delay_ms=0):
        self._started = False
        self._export_dir = export_dir
        self._verbose = verbose
        self._delay_ms = delay_ms

    def start(self):
        if not self._started and self._export_dir is not None:
            try:
                profiler_opts = tf.profiler.experimental.ProfilerOptions(
                    # Ajust TraceMe levels:
                    # - 1: critical
                    # - 2: info [default]
                    # - 3: verbose
                    host_tracer_level=3 if self._verbose else 2,
                    # Enables python function call tracing
                    # - 0: disabled [default]
                    # - 1: enabled
                    python_tracer_level=1 if self._verbose else 0,
                    # Adjust device (TPU/GPU) tracer level:
                    # - 0: disabled
                    # - 1: enabled [default]
                    device_tracer_level=1,
                    delay_ms=self._delay_ms
                )
                tf.profiler.experimental.start(
                    logdir=self._export_dir, options=profiler_opts
                )
                logging.info(
                    "[PROFILER] Starting Profiling - Data will be stored in: "
                    f"`{self._export_dir}`"
                )
                self._started = True

            except tf.errors.AlreadyExistsError:
                logging.warning(
                    "[PROFILER] Could not start the profiler. It "
                    "appears to have been previously been started."
                )

    def stop(self):
        if self._started:
            try:
                tf.profiler.experimental.stop()
                logging.info(
                    "[PROFILER] Stopping Profiling - Data has been stored in: "
                    f"`{self._export_dir}`"
                )
            # profiler has already been stopped or not started
            except tf.errors.UnavailableError:
                logging.warning(
                    "[PROFILER] Could not stop the profiler. It "
                    "appears to have been previously been stopped."
                )
                pass
            self._started = False

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()


@contextlib.contextmanager
def time_and_trace_ctx(message, step_num=None, _r=None):
    with timed_section(message):
        with tf.profiler.experimental.Trace(message, step_num=step_num, _r=_r):
            yield
