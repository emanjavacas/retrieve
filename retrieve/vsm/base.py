
import inspect

import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels

from retrieve.compare import soft_cosine


def init_sklearn_vectorizer(vectorizer, vocab=None, **kwargs):
    params = set(inspect.signature(vectorizer).parameters)
    return vectorizer(
        vocabulary=vocab,
        # overwrite default to avoid ignoring input
        token_pattern=r'\S+',
        **{k: v for k, v in kwargs.items() if k in params})


class VSM:
    def __init__(self, *args, **kwargs):
        """
        Dummy constructor, since not all methods require initalization
        """
        pass

    def fit(self, *args, **kwargs):
        """
        Dummy fit method (not compulsory)
        """
        return self

    def get_similarities(self, queries, index, metric='cosine', **kwargs):
        index, queries = list(index), list(queries)
        transform = self.transform(queries + index)
        queries, index = transform[:len(queries)], transform[len(queries):]
        S = pairwise_kernels(queries, index, metric=metric, jobs=-1, **kwargs)

        return S


class VSMSoftCosine(VSM):
    def __init__(self, vocab, vectorizer, **kwargs):
        self.vectorizer = init_sklearn_vectorizer(vectorizer, vocab=vocab, **kwargs)

    def get_soft_cosine_S(self, queries, index, embs, beta=1, S_metric='cosine'):
        index, queries = list(index), list(queries)
        transform = self.transform(index + queries)
        index, queries = transform[:len(index)], transform[:len(index)]
        S = embs.get_S(words=self.vectorizer.get_feature_names(), metric=S_metric)
        # TODO: apply some kind of relu
        M = np.power(np.clip(S, a_min=0, a_max=np.max(S)), beta)

        return soft_cosine(queries, index, M)
