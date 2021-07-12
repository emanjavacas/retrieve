
# for SIF, see: https://github.com/PrincetonML/SIF

import inspect

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .base import VSM


def remove_pc(W, npc=1):
    pc = TruncatedSVD(n_components=npc, n_iter=7).fit(W).components_
    if npc == 1:
        return W - W.dot(pc.transpose()) * pc
    else:
        return W - W.dot(pc.transpose()).dot(pc)


class SIF(VSM):
    def __init__(self, embs, freqs, npc=1, a=1e-3, **kwargs):
        self.npc = npc
        self.a = a
        # attributes
        self.embs = embs
        self.freqs = freqs
        self.N = sum(freqs.values())

    def get_weight(self, w):
        if w not in self.freqs:
            raise KeyError(w)
        return self.a / (self.a + self.freqs[w] / self.N)

    def transform(self, sents):
        # get average
        output = np.array([
            sum(self.get_weight(w) * self.embs[w] for w in sent) / len(sent)
            for sent in sents])
        output = remove_pc(output, npc=self.npc)
        return output


class TfidfEmbedding(VSM):
    def __init__(self, embs, sents, **kwargs):
        # attributes
        self.embs = embs
        params = set(inspect.signature(TfidfVectorizer).parameters)
        self.tfidf = TfidfVectorizer(
            # overwrite default to avoid ignoring input
            token_pattern=r'\S+',
            **{k: v for k, v in kwargs.items() if k in params})

    def fit(self, sents):
        self.tfidf.fit(' '.join(s) for s in sents)
        return self

    def transform(self, sents):
        X = self.tfidf.transform(sents)
        output = np.zeros((len(sents), ))
        idx2w = self.tfidf.get_feature_names()
        for s_idx in range(len(sents)):
            output[s_idx] = sum(
                weight * self.embs[idx2w[idx]]
                for (_, idx), weight in X[s_idx].todok().items()
            ) / len(sents[s_idx])

        return output


class BOWEmbedding(VSM):
    def __init__(self, embs, **kwargs):
        # attributes
        self.embs = embs

    def transform(self, sents):
        return np.array([sum(self.embs[w] for w in s) / len(s) for s in sents])
