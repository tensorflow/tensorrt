#!/usr/bin/env python
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# -*- coding: utf-8 -*-

import inspect
import os
import subprocess
import shlex

# The `__version__` number shall be updated everytime core benchmarking files
# are updated.
# Please update CHANGELOG.md with a description of what this version changed.
__version__ = "2.0.3"


def get_commit_id():

    try:
        currentdir = os.path.join(
            "/",
            *os.path.dirname(
                os.path.abspath(inspect.getfile(inspect.currentframe()))
            ).split("/")[:-2]
        )

        command_1 = f"git config --global --add safe.directory {currentdir}"
        command_2 = "git rev-parse --short HEAD"

        subprocess.check_output(shlex.split(command_1))
        commit_id = subprocess.check_output(shlex.split(command_2))
        return commit_id.decode('ascii').strip()

    except Exception as e:
        return str(e)
