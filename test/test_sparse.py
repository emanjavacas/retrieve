
import unittest

import numpy as np
import scipy.sparse

from retrieve.corpora import load_vulgate
from retrieve.data import Criterion, TextPreprocessor, FeatureSelector
from retrieve import sparse_utils
from retrieve.methods.set_similarity import SetSimilarity


class TestSparse(unittest.TestCase):
    def setUp(self):
        old, new = load_vulgate(split_testaments=True)
        TextPreprocessor().process_collections(old, new, min_n=2, max_n=4)
        FeatureSelector(old, new).filter_collections(
            old, new, criterion=(Criterion.DF >= 2) & (Criterion.FREQ >= 5))
        # speed up testing
        self.old = old.get_features(cast=set)[:1000]
        self.new = new.get_features(cast=set)[:1000]

    def test_threshold(self):
        for fn in ["containment", "jaccard", "containment_min"]:
            sims = SetSimilarity(0.2, fn).get_similarities(self.new, self.old)
            for th in np.linspace(0.2, 1, 100):
                x1, y1, _ = scipy.sparse.find(sims >= th)
                sims = sparse_utils.set_threshold(sims, th)
                x2, y2, _ = scipy.sparse.find(sims)
                self.assertEqual(x1.shape, x2.shape)
                self.assertEqual(y1.shape, y2.shape)
                self.assertTrue(np.alltrue(x1 == x2))
                self.assertTrue(np.alltrue(y1 == y2))
