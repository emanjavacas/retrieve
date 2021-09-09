
"""
Inspired by:
 - https://github.com/biocore/scikit-bio/blob/master/skbio/alignment/_pairwise.py
 - https://github.com/lingpy/lingpy/blob/master/lingpy/algorithm/cython/calign.pyx
"""

import logging
import collections

import itertools
from itertools import product

import numba as nb
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from ... import _align

logger = logging.getLogger(__name__)

VGAP, HGAP, END, MATCH, UNPROCESSED = range(1, 6)
GAP_SYM = -1


class BaseScorer:
    def get_scores(self, s1, s2):
        raise NotImplementedError

    def get_maximum_score(self, s1, s2):
        raise NotImplementedError
    
    def fit(self, coll1, coll2, embs=None):
        return self


def _get_constant_scores(s1, s2, match, mismatch):
    scores = np.zeros((len(s1), len(s2)))
    for (i, a), (j, b) in product(enumerate(s1), enumerate(s2)):
        scores[i, j] = match[i] if a == b else mismatch
    return scores


class ConstantScorer(BaseScorer):
    def __init__(self, match=2, mismatch=-1, **kwargs):
        self.match = match
        self.mismatch = mismatch

    def get_scores(self, s1, s2):
        match = len(s1) * [self.match]
        return _get_constant_scores(s1, s2, match, self.mismatch)

    def get_maximum_score(self, s1, s2):
        return self.match * min(len(s1), len(s2))


class LookupScorer(BaseScorer):
    def __init__(self, match=2, mismatch=-1, offset_range=(-2, 0), offset_smoothing=1.5, 
            **kwargs):
        self.match = match
        self.mismatch = mismatch
        self.offset_min, self.offset_max = offset_range
        self.offset_smoothing = offset_smoothing
        self.lookup = {}

    def get_scores(self, s1, s2):
        # matches can depend on the actual word (e.g. stop-words)
        match = [self.match + self.lookup.get(w, 0) for w in s1]
        return _get_constant_scores(s1, s2, match, self.mismatch)

    def get_maximum_score(self, s1, s2):
        if len(s1) < len(s2):
            return sum([self.match + self.lookup.get(w, 0) for w in s1])
        else:
            return sum([self.match + self.lookup.get(w, 0) for w in s2])

    def fit(self, coll1, coll2, **kwargs):
        vocab, counts = zip(*collections.Counter(
            [w for coll in [coll1, coll2] for feats in coll.get_features() for w in feats]
        ).most_common())
        counts = np.array(counts)
        scaled = MinMaxScaler(
            feature_range=(self.offset_min, self.offset_max)
        ).fit_transform(np.power(counts, 1/self.offset_smoothing).reshape(-1, 1)).reshape(-1)
        self.lookup = {w: (-v + self.offset_min + self.offset_max) for w, v in zip(vocab, scaled)}
        return self


def _get_embedding_scores(s1, s2, d, S, match, mismatch):
    scores = np.zeros((len(s1), len(s2)))
    for (i, a), (j, b) in product(enumerate(s1), enumerate(s2)):
        if a == b:
            scores[i, j] = match
        elif a in d and b in d:
            scores[i, j] = match * S[d[a], d[b]]
        else:
            scores[i, j] = mismatch
    return scores


class EmbeddingScorer(BaseScorer):
    def __init__(self, match=2, mismatch=-1, field='lemma', **kwargs):
        self.match = match
        self.mismatch = mismatch
        self.S = self.d = self.max_s = None
        self.field = field
        self.kwargs = kwargs

    def get_scores(self, s1, s2):
        return _get_embedding_scores(s1, s2, self.d, self.S, self.match, self.mismatch)

    def get_maximum_score(self, s1, s2):
        return self.match * self.max_s * min(len(s1), len(s2))

    def fit(self, coll1, coll2, embs):
        if 'fill_missing' in self.kwargs:
            raise ValueError("`fill_missing` not allowed for EmbeddingScorer")
        
        from retrieve.data import get_vocab_from_colls
        from retrieve.embeddings import Embeddings
        vocab = get_vocab_from_colls(coll1, coll2, field=self.field)
        embs = Embeddings.require_embeddings(embs, vocab=vocab)
        d, S = embs.get_S(vocab=vocab, **self.kwargs)
        self.d = d
        self.max_s = S.max()

        return self


@nb.njit
def compute_input_matrices(
        s1len, s2len, scores,
        extend_gap=-1, open_gap=-1, terminal_gap=0):
    smatrix = np.zeros((s1len + 1, s2len + 1))
    tmatrix = np.zeros((s1len + 1, s2len + 1), dtype=np.int64) + END
    best_score = 0.0
    row = col = None

    for i in range(1, s1len + 1):
        for j in range(1, s2len + 1):
            mscore = smatrix[i-1, j-1] + scores[i-1, j-1]

            # vertical gap
            if i == s1len:
                # don't penalize vertical gaps if we are finished with s1
                vscore = smatrix[i-1, j] + terminal_gap
            elif tmatrix[i-1, j] == VGAP:
                vscore = smatrix[i-1, j] + extend_gap
            else:
                vscore = smatrix[i-1, j] + open_gap

            # horizontal gap
            if j == s2len:
                # don't penalize horizontal gaps if we are finished with s2
                hscore = smatrix[i, j-1] + terminal_gap
            elif tmatrix[i, j-1] == HGAP:
                hscore = smatrix[i, j-1] + extend_gap
            else:
                hscore = smatrix[i, j-1] + open_gap

            # pick best move
            score, move = 0.0, 0
            for cscore, cmove in [(0.0, END),
                                  (hscore, HGAP),
                                  (vscore, VGAP),
                                  (mscore, MATCH)]:
                if cscore >= score:
                    score, move = cscore, cmove

            smatrix[i, j], tmatrix[i, j] = score, move

            if score >= best_score:
                best_score = score
                row, col = i, j

    return smatrix, tmatrix, row, col, best_score


@nb.njit
def traceback(smatrix, tmatrix, row, col, gap_sym=GAP_SYM):
    a1, a2 = [], []
    move = tmatrix[row, col]

    while move != END:
        if move == MATCH:
            a1.append(row - 1)
            a2.append(col - 1)
            row -= 1
            col -= 1
        elif move == VGAP:
            a1.append(row - 1)
            a2.append(gap_sym)
            row -= 1
        else:
            a1.append(gap_sym)
            a2.append(col - 1)
            col -= 1
        move = tmatrix[row, col]

    return a1[::-1], a2[::-1]


def get_alignment_string(s1, s2, a1, a2, gap_sym=GAP_SYM):
    return ([s1[idx] if idx != gap_sym else gap_sym for idx in a1],
            [s2[idx] if idx != gap_sym else gap_sym for idx in a2])


def get_alignment_ranges(a1, a2, gap_sym=GAP_SYM):

    def _get_alignment_ranges(alignment, gap_sym):
        ranges = []
        for key, group in itertools.groupby(alignment, lambda item: item == gap_sym):
            if key:
                continue
            group = list(group)
            if len(group) > 1:
                start, *_, end = group
            else:
                start = end = group[0]
            ranges.append((start, end + 1))
        return ranges

    return _get_alignment_ranges(a1, gap_sym), _get_alignment_ranges(a2, gap_sym)


def get_printable_string(s1, s2, a1, a2, gap_sym=GAP_SYM):
    a1s, a2s = get_alignment_string(s1, s2, a1, a2, gap_sym=gap_sym)
    assert len(a1s) == len(a2s)

    max1, max2 = max(map(len, a1s)), max(map(len, a2s))
    lines = []
    for a1_i, a2_i in zip(a1s, a2s):
        if a1_i == a2_i:
            sep = '=='
        elif a1_i == gap_sym or a2_i == gap_sym:
            sep = '-'
        else:
            sep = '!='
        lines.append("{} {} {}".format(
            a1_i.ljust(max1),
            sep,
            a2_i.ljust(max2)))

    return '\n'.join(lines)


def get_horizontal_alignment(s1, s2, a1=None, a2=None, gap_sym=GAP_SYM, **kwargs):
    if a1 is None or a2 is None:
        a1, a2, _ = local_alignment(s1, s2, **kwargs)

    # empty alignment
    if not a1 or not a2 or len(s1) < 2:
        return ' '.join(s1), ' ', ' '.join(s2)

    # find start of both seqs
    str1 = str2 = alignment = ''
    start1 = next(filter(lambda it: it != gap_sym, a1))
    start2 = next(filter(lambda it: it != gap_sym, a2))
    str1 += ' '.join(s1[:start1])
    str2 += ' '.join(s2[:start2])
    str1, str2 = str1.rjust(len(str2)), str2.rjust(len(str1))
    alignment += ' ' * len(str1)
    for w1, w2 in zip(a1, a2):
        if w1 == gap_sym:
            str1 += ' ' + (' ' * len(s2[w2]))
            str2 += ' ' + s2[w2]
            alignment += ' ' + (' ' * len(s2[w2]))
        elif w2 == gap_sym:
            str1 += ' ' + s1[w1]
            str2 += ' ' + (' ' * len(s1[w1]))
            alignment += ' ' + (' ' * len(s1[w1]))
        else:
            str1 += ' ' + s1[w1].rjust(len(s2[w2]))
            str2 += ' ' + s2[w2].rjust(len(s1[w1]))
            sym = ('-' if s1[w1] == s2[w2] else 'x')
            alignment += ' ' + (sym * len(s2[w2])).rjust(len(s1[w1]))

    # attach end of both seqs
    end1 = next(filter(lambda it: it != gap_sym, reversed(a1))) + 1
    end2 = next(filter(lambda it: it != gap_sym, reversed(a2))) + 1
    end1, end2 = ' '.join(s1[end1:]), ' '.join(s2[end2:])
    str1 += ' ' + end1.ljust(len(end2))
    str2 += ' ' + end2.ljust(len(end1))

    return str1, alignment, str2


def get_local_alignment_numba(
        s1, s2, scorer, extend_gap, open_gap, terminal_gap, only_score):
    scores = scorer.get_scores(list(s1), list(s2))
    smatrix, tmatrix, row, col, score = compute_input_matrices(
        len(s1), len(s2), scores, extend_gap, open_gap, terminal_gap)
    if only_score:
        return score
    a1, a2 = traceback(smatrix, tmatrix, row, col)
    return a1, a2, score


def get_local_alignment_cython(
        s1, s2, scorer, extend_gap, open_gap, terminal_gap, only_score):
    scores = scorer.get_scores(list(s1), list(s2))
    return _align.sw_alignment(
        len(s1), len(s2), extend_gap, open_gap, terminal_gap, scores, only_score)


def local_alignment(s1, s2, scorer=ConstantScorer(),
                    extend_gap=-1, open_gap=-1, terminal_gap=0,
                    impl='cython', only_score=False, normalize=False):
    if len(s1) == 0 or len(s2) == 0:
        if only_score:
            return 0.0
        return [], [], 0.0

    if impl == 'cython' and _align is not None:
        fn = get_local_alignment_cython
    else:
        fn = get_local_alignment_numba

    res = fn(s1, s2, scorer, extend_gap, open_gap, terminal_gap, only_score)

    if normalize:
        max_score = scorer.get_maximum_score(s1, s2)
        if only_score:
            return res / max_score
        else:
            a, b, score = res
            return a, b, score / max_score

    return res
