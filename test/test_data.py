
import unittest

from retrieve.data import TextPreprocessor, FeatureSelector, Criterion
from retrieve import utils

from retrieve.corpora import load_vulgate, load_alice


class TestCriterion(unittest.TestCase):
    def setUp(self):
        collection = load_vulgate()
        stops = utils.Stopwords('latin.stop')
        processor = TextPreprocessor(
            stopwords=stops, 
            field_regexes={'token': {'should_match': True, 'regex': '[a-z]+'}})
        processor.process_collections(collection)
        self.fsel = FeatureSelector(collection)

    def test_threshold(self):
        for th_min, th_max in zip(range(1, 1000, 100), range(100, 10000, 1000)):
            vocab = self.fsel.get_vocab(th_min <= Criterion.DF < th_max)
            for ft in vocab:
                self.assertTrue(self.fsel.dfs[self.fsel.features[ft]] >= th_min)
                self.assertTrue(self.fsel.dfs[self.fsel.features[ft]] < th_max)


class TestCriterionShingling(unittest.TestCase):
    def setUp(self):
        collection = load_alice()
        processor = TextPreprocessor()
        processor.process_collections(collection)
        self.fsel = FeatureSelector(collection)

    def test_threshold(self):
        for th_min, th_max in zip(range(1, 1000, 100), range(100, 10000, 1000)):
            vocab = self.fsel.get_vocab(th_min <= Criterion.DF < th_max)
            for ft in vocab:
                self.assertTrue(self.fsel.dfs[self.fsel.features[ft]] >= th_min)
                self.assertTrue(self.fsel.dfs[self.fsel.features[ft]] < th_max)
