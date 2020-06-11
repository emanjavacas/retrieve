
import collections
import itertools
import json

from retrieve.data import Doc, Ref, Collection


def decode_ref(ref):
    book, chapter, verse = ref.split('_')
    book = ' '.join(book.split('-'))
    return book, chapter, verse


def encode_ref(ref):
    book, chapter, verse = ref
    return '-'.join(book.split()) + '_' + '_'.join([chapter, verse])


def read_testament_books(testament='new'):
    books = []
    with open('texts/{}-testament.books'.format(testament)) as f:
        for line in f:
            line = line.strip()
            books.append(line)
    return books


def load_vulgate(path='texts/vulgate.csv',
                 include_blb=False, split_testaments=False,
                 max_verses=-1):

    blb = collections.defaultdict(list)
    if include_blb:
        for ref in load_blb_refs():
            blb[ref['source']].append(ref)

    with open(path) as f:
        docs = []
        for idx, l in enumerate(f):
            if not l.strip():
                continue
            if max_verses > 0 and len(docs) >= max_verses:
                break
            book, chapter, verse, tokens, pos, tt, pie = l.strip().split('\t')
            doc_id = book, chapter, verse
            refs = [Ref(ref['target'], meta=ref['ref_type']) for ref in blb[doc_id]]
            fields = {'token': tokens.split(), 'lemma': pie.split(), 'tt': tt.split()}
            docs.append(Doc(fields=fields, refs=refs, doc_id=doc_id))

    if split_testaments:
        old, new = set(read_testament_books('old')), set(read_testament_books('new'))
        old_books, new_books = [], []
        for doc in docs:
            if doc.doc_id[0] in old:
                old_books.append(doc)
            elif doc.doc_id[0] in new:
                new_books.append(doc)

        return Collection(old_books), Collection(new_books)

    return Collection(docs)


def is_range(refs):
    book, chapter, verse = zip(*refs)
    if len(set(book)) == 1 and len(set(chapter)) == 1:
        verse = list(sorted(map(int, verse)))
        start, *_, stop = verse
        if start != stop and list(range(start, stop + 1)) == verse:
            return True
    return False


def load_blb_refs(path='texts/blb.refs.json', max_range=5):
    with open(path) as f:
        refs = []
        for ref in json.loads(f.read()):
            ref['source'] = decode_ref(ref['source'])
            ref['target'] = decode_ref(ref['target'])
            refs.append(ref)

    def key(obj):
        return obj.get('group')

    filtered = []
    refs.sort(key=key)
    for _, group in itertools.groupby(refs, key=key):
        group = list(group)
        if len(group) > 1 and max_range > 0 and len(group) > max_range:
            if is_range([ref['source'] for ref in group]):
                continue
            elif is_range([ref['target'] for ref in group]):
                continue
        filtered.extend(group)

    return filtered
