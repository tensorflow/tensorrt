#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

from contextlib import contextmanager
from glob import glob, iglob

__all__ = [
    "CustomTestCase",
    "list_all_py_files",
]


class CustomTestCase(unittest.TestCase):

    @contextmanager
    def assertNotRaises(self, exc_type):
        try:
            yield None
        except exc_type:
            raise self.failureException(f"{exc_type.__name__} raised")


_excludes_paths = ["tftrt/blog_posts/"]


def list_all_py_files():
    for _dir in ["tests", os.path.join("tftrt", "benchmarking-python")]:
        for _file in iglob(f"{_dir}/**/*.py", recursive=True):
            if any([path in _file for path in _excludes_paths]):
                continue
            yield _file
