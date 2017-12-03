from __future__ import absolute_import

import fnmatch
import os
import unittest


def find_files(directory, pattern='*.py'):
    for root, _, files in os.walk(directory):
        for filename in files:
            if fnmatch.fnmatch(filename, pattern):
                filename = os.path.abspath(os.path.join(root, filename))
                yield filename


class TestExamples(unittest.TestCase):
    def test_examples(self):
        for filename in find_files(os.path.join(os.path.dirname(__file__), os.path.pardir, 'examples')):
            exec(open(filename).read(), globals())
