
import unittest

import random

from retrieve.methods import set_similarity
from retrieve.methods import SetSimilarity
from retrieve.corpora import load_vulgate


def _test_similarity_func(threshold, func, old, new):
    return SetSimilarity(threshold, func).get_similarities(
        old.get_features(cast=set, field='lemma'),
        new.get_features(cast=set, field='lemma'),
        processes=5)


class TestSet(unittest.TestCase):
    def setUp(self):
        old, new = load_vulgate(split_testaments=True)
        self.old = old
        self.new = new

    def _test_similarities(self, sims, n_samples, func):
        for (i, j), sim in random.sample(sims.todok().items(), n_samples):
            old, new = self.old[i], self.new[j]
            expected = getattr(old, func)(new, field='lemma')
            self.assertAlmostEqual(
                expected, sim,
                msg="{}:{}, {} expected: {}, but got {}".format(
                    i, j, func, expected, sim))

    def test_containment(self):
        self._test_similarities(
            _test_similarity_func(0.5, 'containment', self.old, self.new),
            500, 'containment')

    def test_jaccard(self):
        self._test_similarities(
            _test_similarity_func(0.35, 'jaccard', self.old, self.new),
            500, 'jaccard')

    def test_containment_min(self):
        self._test_similarities(
            _test_similarity_func(0.5, 'containment_min', self.old, self.new),
            500, 'containment_min')
