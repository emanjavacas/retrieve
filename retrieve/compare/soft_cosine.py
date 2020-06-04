
import numpy as np

import tqdm


def soft_cosine4(queries, index, S):
    """
    This function assumes that the order of vocabulary in the similarity matrix
    correspondes to the order of the vocabulary in the document representations

    Arguments
    =========

    queries : np.array(n, vocab), n query docs in BOW format
    index : np.array(m, vocab), m indexed docs in BOW format
    S : np.array(vocab, vocab), similarity matrix (possibly raised to a power)

    Output
    ======
    sims : np.array(n, m), soft cosine similarities
    """
    sims = np.zeros((len(queries), len(index)))
    SindexT = S @ index.T
    den2 = np.sqrt(np.diag(index @ SindexT))
    for idx, s in tqdm.tqdm(enumerate(queries), total=len(queries), desc='Soft cosine'):
        num = s[None, :] @ SindexT
        den1 = np.sqrt(s @ S @ s)
        sims[idx] = (num / ((np.ones(len(den2)) * den1) * den2))[0]
    return np.nan_to_num(sims, copy=False)
