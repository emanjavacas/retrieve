
import unittest

import random

from retrieve.methods import set_similarity
from retrieve.methods import SetSimilarity
from retrieve.corpora import load_vulgate


def _test_similarity_func(threshold, func, coll):
    return SetSimilarity(threshold, func).get_similarities(
        coll.get_features(cast=set, field='lemma'),
        coll.get_features(cast=set, field='lemma'),
        processes=5)


class TestSet(unittest.TestCase):
    def setUp(self):
        self.vulgate = load_vulgate(max_verses=1000)

    def _test_similarities(self, sims, n_samples, func):
        for (i, j), sim in random.sample(sims.todok().items(), n_samples):
            vul_a, vul_b = self.vulgate[i], self.vulgate[j]
            expected = getattr(vul_a, func)(vul_b, field='lemma')
            self.assertAlmostEqual(
                expected, sim,
                msg="{}:{}, {} expected: {}, but got {}".format(
                    i, j, func, expected, sim))

    def test_containment(self):
        self._test_similarities(
            _test_similarity_func(0.5, 'containment', self.vulgate),
            500, 'containment')

    def test_jaccard(self):
        self._test_similarities(
            _test_similarity_func(0.35, 'jaccard', self.vulgate),
            500, 'jaccard')

    def test_containment_min(self):
        self._test_similarities(
            _test_similarity_func(0.5, 'containment_min', self.vulgate),
            500, 'containment_min')

    def test_cosine(self):
        self._test_similarities(
            _test_similarity_func(0.5, 'cosine', self.vulgate),
            500, 'cosine')
