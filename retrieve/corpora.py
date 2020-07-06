
import os
import glob
import json
import collections
import warnings

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
    with open('data/texts/{}-testament.books'.format(testament)) as f:
        for line in f:
            line = line.strip()
            books.append(line)
    return books


def read_bible(path, fields=('token', 'pos', '_', 'lemma'), max_verses=-1,
               sort_docs=True):
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
            try:
                docs.append(Doc(fields=dict(zip(fields, data)), doc_id=doc_id))
            except ValueError:
                warnings.warn("Ignoring document {}".format(' '.join(doc_id)))

    if not sort_docs:
        return docs

    books = {book: idx for idx, book in enumerate(
        read_testament_books('old') + read_testament_books('new'))}

    def key(doc):
        book, chapter, verse = doc.doc_id
        return books.get(book, len(books)), int(chapter), int(verse)

    return sorted(docs, key=key)


def load_vulgate(path='data/texts/vulgate.csv',
                 include_blb=False, split_testaments=False, **kwargs):

    docs = read_bible(path, **kwargs)
    coll = Collection(docs, name=os.path.basename(path))
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
            warnings.warn("Missing book: {}".format(doc.doc_id[0]))

    # add refs
    old = Collection(old_books, name='Old Testament')
    new = Collection(new_books, name='New Testament')
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


def load_blb_refs(path='data/texts/blb.refs.json'):
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
        output = []
        for ref in json.load(f):
            ref['target'] = list(map(tuple, ref['target']))
            output.append(ref)
        return output


def shingle_doc(doc, f_id, overlap=10, window=20):
    shingled_docs = []
    for start in range(0, len(next(iter(doc.values()))), window - overlap):
        # doc might be smaller than window
        stop = min(start + window, len(doc['token'][start:]))
        # doc id
        doc_id = f_id, (start, stop)
        # prepare doc
        fields = {key: vals[start:stop] for key, vals in doc.items()}
        fields['ids'] = list(range(start, stop))

        shingled_docs.append(Doc(fields=fields, doc_id=doc_id))

    return shingled_docs


def load_bernard(directory='data/texts/bernard',
                 bible_path='data/texts/vulgate.csv',
                 **kwargs):

    bible = Collection(read_bible(bible_path), name='Vulgate')
    shingled_docs, shingled_refs = [], []
    for path in glob.glob(os.path.join(directory, '*.txt')):
        refs = read_refs(path.replace('.txt', '.refs.json'))
        doc = read_doc(path, fields=('w_id', 'token', 'pos', '_', 'lemma'))
        shingles = shingle_doc(doc, path, **kwargs)
        for ref in refs:
            source = []
            for shingle in shingles:
                if set(ref['source']).intersection(set(shingle.fields['ids'])):
                    source.append(shingle.doc_id)
            if source:
                target = ref['target']
                shingled_refs.append(Ref(tuple(source), tuple(target), meta=ref))

        shingled_docs.extend(shingles)

    shingled_docs = Collection(shingled_docs, name='Bernard')

    return shingled_docs, bible, shingled_refs
