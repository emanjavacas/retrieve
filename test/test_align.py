

import unittest
import multiprocessing as mp

from retrieve import utils
from retrieve.methods import local_alignment, align_collections
from retrieve.corpora import load_vulgate


class TestAlign(unittest.TestCase):
    def test_numba(self):
        s1 = 'AGCACACA'
        s2 = 'ACACACTA'
        self.assertEqual(local_alignment(s1, s2), local_alignment(s1, s2, impl='numba'))

        s1 = "ATAGACGACATACAGACAGCATACAGACAGCATACAGA"
        s2 = "TTTAGCATGCGCATATCAGCAATACAGACAGATACG"
        self.assertEqual(local_alignment(s1, s2), local_alignment(s1, s2, impl='numba'))

    def test_parallel(self):
        coll = load_vulgate(max_verses=500)
        sims_single = align_collections(coll, processes=1, field='lemma')
        cpus = mp.cpu_count()
        with utils.timer() as timer:
            sims_parallel = align_collections(coll, processes=cpus, field='lemma')
            timer(desc='Parallel alignment on {} CPUs'.format(cpus))
        self.assertTrue(all(sims_single.data == sims_parallel.data))
