#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import unittest

import pygments
from pygments import console

from tests.utils import list_all_py_files
from tests.utils import CustomTestCase

from yapf.yapflib.yapf_api import FormatCode


def _read_utf_8_file(filename):
    if sys.version_info.major == 2:  ## Python 2 specific
        with open(filename, 'rb') as f:
            return unicode(f.read(), 'utf-8')
    else:
        with open(filename, encoding='utf-8') as f:
            return f.read()


def print_color(msg, color):
    print(pygments.console.colorize(color, msg))


class YAPF_Style_Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.badly_formatted_files = list()
        cls.files_2_test = list_all_py_files()

    def test_files_format(self):

        total_analyzed_files = 0
        for file in list_all_py_files():

            total_analyzed_files += 1

            try:

                print(f"Testing: {file:100s}", end="")
                code = _read_utf_8_file(file)

                # https://pypi.python.org/pypi/yapf/0.20.2#example-as-a-module
                diff, changed = FormatCode(
                    code,
                    filename=file,
                    style_config='.style.yapf',
                    print_diff=True
                )

                if changed:
                    print_color("FAILURE", "red")
                    self.badly_formatted_files.append(file)
                else:
                    print_color("SUCCESS", "green")

            except Exception as e:
                print_color("FAILURE", "red")("FAILURE")
                print(
                    "Error while processing file: `%s`\n"
                    "Error: %s" % (file, str(e))
                )

        str_err = ""

        if self.badly_formatted_files:
            for filename in self.badly_formatted_files:
                str_err += f"yapf -i --style=.style.yapf {filename}\n"

            str_err = "\n======================================================================================\n" \
                        f"Bad Coding Style: {len(self.badly_formatted_files)} file(s) need to be formatted, run the following commands to fix: \n" \
                        f"{str_err}" \
                        "======================================================================================"

        passing_files = total_analyzed_files - len(self.badly_formatted_files)
        print_color(
            f"\nPASSING: {passing_files} / {total_analyzed_files}",
            "green" if str_err == "" else "red"
        )

        if str_err != "":
            print_color(str_err, "red")

        self.assertEqual(str_err, "")


if __name__ == '__main__':
    unittest.main()
