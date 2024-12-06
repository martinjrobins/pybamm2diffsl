import unittest

import numpy as np

from pybamm2diffsl.diffeq import Diffeq
from tests.logistic import logistic


class TestVector(unittest.TestCase):
    def setUp(self):
        self.diffeq = Diffeq(logistic)

    def test_create(self):
        v = self.diffeq.vector([1, 2, 3])
        contents = v.getFloat64Array()
        np.testing.assert_array_equal(contents, [1, 2, 3])
        v.destroy()

    def test_edit(self):
        v = self.diffeq.vector([1, 2, 3])
        v.resize(5)
        contents = v.getFloat64Array()
        contents[3] = 4
        contents[4] = 0
        contents[0] = -1
        np.testing.assert_array_equal(contents, [-1, 2, 3, 4, 0])
        v.destroy()
