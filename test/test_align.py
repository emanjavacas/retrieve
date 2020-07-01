
import unittest

from retrieve.methods import local_alignment


class TestAlign(unittest.TestCase):
    def test_numba(self):
        s1 = 'AGCACACA'
        s2 = 'ACACACTA'
        self.assertEqual(local_alignment(s1, s2), local_alignment(s1, s2, impl='numba'))

        s1 = "ATAGACGACATACAGACAGCATACAGACAGCATACAGA"
        s2 = "TTTAGCATGCGCATATCAGCAATACAGACAGATACG"
        self.assertEqual(local_alignment(s1, s2), local_alignment(s1, s2, impl='numba'))
