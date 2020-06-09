
import unittest

import numpy as np
import scipy.sparse

from retrieve.corpora import load_vulgate
from retrieve.data import Criterion, TextPreprocessor, FeatureSelector
from retrieve import utils
from retrieve.vsm.lexical import Tfidf

from retrieve.compare import soft_cosine_similarities, soft_cosine_simple

from sklearn.metrics import pairwise_kernels


class TestSoftCosine(unittest.TestCase):
    def setUp(self):
        # load
        vulg = load_vulgate(include_blb=True, max_verses=1000)
        # preprocess
        TextPreprocessor().process_collection(vulg, min_n=1, max_n=1)
        # drop features and get vocabulary
        fsel = FeatureSelector(vulg)
        vocab = fsel.filter_collection(vulg, (Criterion.DF >= 2) & (Criterion.FREQ >= 5))
        # get documents
        feats, _ = vulg.get_nonempty_features(cast=set)
        # transform to tfidf
        feats = Tfidf(vocab).fit(feats).transform(feats)
        query, index = feats[:feats.shape[0]//2], feats[feats.shape[0]//2:]
        # load embeddings, make sure S is in same order as vocab
        embs = utils.Embeddings.from_csv('latin.lemma.embeddings', vocab=vocab)

        self.embs = embs
        self.query = query
        self.index = index
        self.vocab = vocab

    def test_degenerate(self):
        # simple cosine similarity (we always return normalized vectors)
        sims1 = pairwise_kernels(self.query, self.index, metric='linear')
        # degenerate soft cosine should be equal to cosine
        sims2 = soft_cosine_similarities(
            self.query, self.index, np.identity(len(self.vocab)))

        self.assertTrue(np.allclose(sims1, sims2))

    def test_full(self):
        S = self.embs.get_S(words=self.vocab, fill_missing=True, cutoff=0.75, beta=2)
        sims = soft_cosine_similarities(self.query, self.index, S)
        for i in range(10):
            for j in range(10):
                sim1 = sims[i, j]
                sim2 = soft_cosine_simple(self.query[i], self.index[j], S)
                self.assertAlmostEqual(sim1, sim2, msg=f"{i}!={j}")

    def test_sparse(self):
        S = self.embs.get_S(words=self.vocab, fill_missing=True, cutoff=0.0, beta=2)
        sims1 = soft_cosine_similarities(self.query, self.index, S)
        self.assertFalse(scipy.sparse.issparse(sims1))
        S = scipy.sparse.lil_matrix(S)
        sims2 = soft_cosine_similarities(self.query, self.index, S)
        self.assertTrue(scipy.sparse.issparse(sims2))
        self.assertTrue(np.allclose(sims2.todense(), sims1.todense))
