
import os
import glob
import json
import collections

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


def read_bible(path, fields=('token', 'pos', '_', 'lemma'), max_verses=-1):
    with open(path) as f:
        docs = []
        for line in f:
            if not line.strip():
                continue
            if max_verses > 0 and len(docs) >= max_verses:
                break
            book, chapter, verse, *data = line.strip().split('\t')
            data = [field.split() for field in data]
            if len(data) != len(fields):
                raise ValueError(
                    "Expected {} metadata fields, but got {}. File: {}"
                    .format(len(fields), len(data), path))
            doc_id = book, chapter, verse
            docs.append(Doc(fields=dict(zip(fields, data)), doc_id=doc_id))

    return docs


def load_vulgate(path='texts/vulgate.csv',
                 include_blb=False, split_testaments=False, **kwargs):

    docs = read_bible(path, **kwargs)
    coll = Collection(docs)
    if not split_testaments:
        return coll

    # split
    old, new = set(read_testament_books('old')), set(read_testament_books('new'))
    old_books, new_books = [], []
    for doc in docs:
        if doc.doc_id[0] in old:
            old_books.append(doc)
        elif doc.doc_id[0] in new:
            new_books.append(doc)
        else:
            raise ValueError("Missing book:", doc.doc_id[0])

    # add refs
    old, new = Collection(old_books), Collection(new_books)
    if not include_blb:
        return old, new

    refs = []
    for ref in load_blb_refs():
        source, target = [], []
        for r in ref['source']:
            if r not in new:
                print("Couldn't find verse: new ", r)
                continue
            source.append(new.get_doc_idx(r))
        for r in ref['target']:
            if r not in old:
                print("Couldn't find verse: old", r)
                continue
            target.append(old.get_doc_idx(r))

        refs.append(Ref(tuple(source), tuple(target),
                        meta={'ref_type': ref['ref_type'],
                              'source': ref['source'],
                              'target': ref['target']}))

    return old, new, refs


def load_blb_refs(path='texts/blb.refs.json'):
    with open(path) as f:
        return [{'source': [decode_ref(s_ref) for s_ref in ref['source']],
                 'target': [decode_ref(t_ref) for t_ref in ref['target']],
                 'ref_type': ref['ref_type']}
                for ref in json.load(f)]


def read_doc(path, fields=('token', 'pos', '_', 'lemma')):
    output = collections.defaultdict(list)
    with open(path) as f:
        for line in f:
            try:
                data = line.strip().split('\t')
                if len(data) != len(fields):
                    raise ValueError(
                        "Expected {} metadata fields, but got {}. File: {}"
                        .format(len(fields), len(data), path))
                for key, val in zip(fields, data):
                    if key != '_':
                        output[key].append(val)
            except ValueError as e:
                raise e
            except Exception:
                print(line)

    return dict(output)


def read_refs(path):
    with open(path) as f:
        return json.load(f)


def shingle_doc(doc, f_id, overlap=10, window=20):
    shingled_docs = []
    for start in range(0, len(next(iter(doc.values()))), window - overlap):
        stop = start + window
        # doc id
        doc_id = f_id, (start, stop)
        # prepare doc
        fields = {key: vals[start:stop] for key, vals in doc.items()}
        fields['ids'] = list(range(start, stop))
        shingled_docs.append(Doc(fields=fields, doc_id=doc_id))

    return shingled_docs


def load_bernard(directory='texts/bernard', bible_path='texts/vulgate.csv', **kwargs):
    bible = Collection(read_bible(bible_path))
    shingled_docs, shingled_refs = [], []
    for path in glob.glob(os.path.join(directory, '*.txt')):
        doc = read_doc(path, fields=('w_id', 'token', 'pos', '_', 'lemma'))
        refs = read_refs(path.replace('.txt', '.refs.json'))
        for r in refs:
            source = []
            for idx, subdoc in enumerate(shingle_doc(doc, path, **kwargs)):
                if set(r['target']).intersection(set(doc.fields['ids'])):
                    source.append(idx)
            if not source:
                print("missing ref")
                continue
            target = [bible.get_doc_idx(decode_ref(v_id)) for v_id in r['target']]
            shingled_refs.append(Ref(tuple(source), tuple(target), meta=r))
            shingled_docs.append(subdoc)

    shingled_docs = Collection(shingled_docs)

    return shingled_docs, bible, shingled_refs
