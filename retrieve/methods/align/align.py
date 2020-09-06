
"""
Inspired by:
 - https://github.com/biocore/scikit-bio/blob/master/skbio/alignment/_pairwise.py
 - https://github.com/lingpy/lingpy/blob/master/lingpy/algorithm/cython/calign.pyx
"""

import logging

import itertools
from itertools import product

import numba as nb
import scipy.sparse
import numpy as np

import pyximport; pyximport.install()  # noqa: E702
from . import _align


logger = logging.getLogger(__name__)

VGAP, HGAP, END, MATCH, UNPROCESSED = range(1, 6)
GAP_SYM = -1


def _get_embedding_scores(s1, s2, S_lut, match, mismatch):
    scores = np.zeros((len(s1), len(s2)))
    for (i, a), (j, b) in product(enumerate(s1), enumerate(s2)):
        if a == b:
            scores[i, j] = match
        elif (a, b) in S_lut:
            scores[i, j] = match * S_lut[a, b]
        else:
            scores[i, j] = mismatch
    return scores


def create_embedding_scorer(embs, match=2, mismatch=-1, **kwargs):
    """
    embs : retrieve.Embeddings object
    """
    _, S = embs.get_S(**kwargs)
    return EmbeddingScorer(list(embs.keys), S, match=match, mismatch=mismatch)


class EmbeddingScorer:
    def __init__(self, d, S, match=2, mismatch=-1):
        self.match = match
        self.mismatch = mismatch

        # construct lookup similarity
        self.S_lut = self.construct_S(d, S)

    def construct_S(self, keys, S):
        S_lut = {}
        # get indices
        if scipy.sparse.issparse(S):
            ii, jj, _ = scipy.sparse.find(S)
        else:
            ii, jj = np.where(S > 0)
        # build lookup
        for i, j in zip(ii, jj):
            if i == j:
                continue
            S_lut[keys[i], keys[j]] = S[i, j]
            S_lut[keys[j], keys[i]] = S[i, j]

        return S_lut

    def get_scores(self, s1, s2):
        return _get_embedding_scores(s1, s2, self.S_lut, self.match, self.mismatch)


def _get_constant_scores(s1, s2, match, mismatch):
    scores = np.zeros((len(s1), len(s2)))
    for (i, a), (j, b) in product(enumerate(s1), enumerate(s2)):
        scores[i, j] = match if a == b else mismatch
    return scores


class ConstantScorer:
    def __init__(self, match=2, mismatch=-1):
        self.match = match
        self.mismatch = mismatch

    def get_scores(self, s1, s2):
        return _get_constant_scores(s1, s2, self.match, self.mismatch)


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
            for cscore, cmove in [(0, END),
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
    if not a1 or not a2:
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
                    impl='cython', only_score=False):
    if len(s1) == 0 or len(s2) == 0:
        if only_score:
            return 0.0
        return [], [], 0.0

    if impl == 'cython':
        fn = get_local_alignment_cython
    else:
        fn = get_local_alignment_numba

    return fn(s1, s2, scorer, extend_gap, open_gap, terminal_gap, only_score)
