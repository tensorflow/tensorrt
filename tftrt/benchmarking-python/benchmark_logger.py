#! /usr/bin/python
# -*- coding: utf-8 -*-

# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

import inspect
import warnings

from contextlib import contextmanager

from six import add_metaclass

import logging as _logging

from benchmark_info import __version__

from logging_utils.formatters import BaseFormatter
from logging_utils.metaclasses import SingletonMetaClass

__all__ = [
    'Logger',
]


class StdOutFormatter(BaseFormatter):
    DEFAULT_FORMAT = f"%(color)s[BENCH - v{__version__}] "
    DEFAULT_FORMAT += "%(levelname)-8s: %(end_color)s%(message)s"


@add_metaclass(SingletonMetaClass)
class Logger(object):

    # Level 0
    NOTSET = _logging.NOTSET

    # Level 10
    DEBUG = _logging.DEBUG

    # Level 20
    INFO = _logging.INFO

    # Level 30
    WARNING = _logging.WARNING

    # Level 40
    ERROR = _logging.ERROR

    # Level 50
    CRITICAL = _logging.CRITICAL

    _level_names = {
        0: 'NOTSET',
        10: 'DEBUG',
        20: 'INFO',
        30: 'WARNING',
        40: 'ERROR',
        50: 'CRITICAL',
    }

    def __init__(self, capture_io=True):

        self._logger = None

        self._handlers = dict()

        self._define_logger()

    def _define_logger(self):

        # Use double-checked locking to avoid taking lock unnecessarily.
        if self._logger is not None:
            return self._logger

        try:
            # Scope the TensorFlow logger to not conflict with users' loggers.
            self._logger = _logging.getLogger('benchmarking_suite')
            self.reset_stream_handler()

        finally:
            self.set_verbosity(verbosity_level=Logger.INFO)

        self._logger.propagate = False

    def reset_stream_handler(self):

        if self._logger is None:
            raise RuntimeError(
                "Impossible to set handlers if the Logger is not predefined"
            )

        # ======== Remove Handler if already existing ========

        try:
            self._logger.removeHandler(self._handlers["stream_stdout"])
        except KeyError:
            pass

        try:
            self._logger.removeHandler(self._handlers["stream_stderr"])
        except KeyError:
            pass

        # ================= Streaming Handler =================

        # Add the output handler.
        self._handlers["stream_stdout"] = _logging.StreamHandler(sys.stdout)
        self._handlers["stream_stdout"].addFilter(
            lambda record: record.levelno <= _logging.INFO
        )

        self._handlers["stream_stderr"] = _logging.StreamHandler(sys.stderr)
        self._handlers["stream_stderr"].addFilter(
            lambda record: record.levelno > _logging.INFO
        )

        Formatter = StdOutFormatter

        self._handlers["stream_stdout"].setFormatter(Formatter())
        self._logger.addHandler(self._handlers["stream_stdout"])

        try:
            self._handlers["stream_stderr"].setFormatter(Formatter())
            self._logger.addHandler(self._handlers["stream_stderr"])
        except KeyError:
            pass

    def get_verbosity(self):
        """Return how much logging output will be produced."""
        if self._logger is not None:
            return self._logger.getEffectiveLevel()

    def set_verbosity(self, verbosity_level):
        """Sets the threshold for what messages will be logged."""
        if self._logger is not None:
            self._logger.setLevel(verbosity_level)

            for handler in self._logger.handlers:
                handler.setLevel(verbosity_level)

    @contextmanager
    def temp_verbosity(self, verbosity_level):
        """Sets the a temporary threshold for what messages will be logged."""

        if self._logger is not None:

            old_verbosity = self.get_verbosity()

            try:
                self.set_verbosity(verbosity_level)
                yield

            finally:
                self.set_verbosity(old_verbosity)

        else:
            try:
                yield

            finally:
                pass

    def debug(self, msg, *args, **kwargs):
        """
        Log 'msg % args' with severity 'DEBUG'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.debug("Houston, we have a %s", "thorny problem", exc_info=1)
        """
        if self._logger is not None:
            self._logger._log(Logger.DEBUG, msg, args, **kwargs)

    def info(self, msg, *args, **kwargs):
        """
        Log 'msg % args' with severity 'INFO'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.info("Houston, we have a %s", "interesting problem", exc_info=1)
        """
        if self._logger is not None:
            self._logger._log(Logger.INFO, msg, args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """
        Log 'msg % args' with severity 'WARNING'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.warning("Houston, we have a %s", "bit of a problem", exc_info=1)
        """
        if self._logger is not None:
            self._logger._log(Logger.WARNING, msg, args, **kwargs)

    def error(self, msg, *args, **kwargs):
        """
        Log 'msg % args' with severity 'ERROR'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.error("Houston, we have a %s", "major problem", exc_info=1)
        """
        if self._logger is not None:
            self._logger._log(Logger.ERROR, msg, args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        """
        Log 'msg % args' with severity 'CRITICAL'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.critical("Houston, we have a %s", "major disaster", exc_info=1)
        """
        if self._logger is not None:
            self._logger._log(Logger.CRITICAL, msg, args, **kwargs)


# Necessary to catch the correct caller
_logging._srcfile = os.path.normcase(inspect.getfile(Logger.__class__))

logging = Logger()
