
import os

import scipy.sparse

import flask
from flask import Flask
from flask import render_template


def extract_matching_words(doc1, doc2):
    doc1ws = [w for feat in doc1.get_features() for w in feat.split('--')]
    doc2ws = [w for feat in doc2.get_features() for w in feat.split('--')]
    intersect = set(doc1ws).intersection(doc2ws)
    doc1ids, doc2ids = [], []
    for idx, lem in enumerate(doc1.fields['lemma']):
        if lem in intersect:
            doc1ids.append(idx)
    for idx, lem in enumerate(doc2.fields['lemma']):
        if lem in intersect:
            doc2ids.append(idx)

    return doc1ids, doc2ids


class VisualizerApp:
    def __init__(self, sims, coll1, coll2=None, sim_range=(None, None),
                 host='localhost', port=5000):
        # data
        if coll2 is None:
            # drop diagonal
            sims = scipy.sparse.tril(sims)

        coll2 = coll2 or coll1
        # force long collection on y axis
        if len(coll1) > len(coll2):
            self.coll2 = coll1
            self.coll1 = coll2
            self.sims = sims.T
        else:
            self.coll1 = coll1
            self.coll2 = coll2
            self.sims = sims

        self.min_sim, self.max_sim = sim_range

        # app
        self.host = host
        self.port = port
        self.app = Flask(
            __name__,
            # this breaks if the app is moved to a different directory
            template_folder=os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'templates'))

        # add rules
        self.app.add_url_rule("/", "index", self.index)
        self.app.add_url_rule("/matching", "matching", self.matching, methods=['GET'])
        self.app.add_url_rule("/heatmap", "heatmap", self.heatmap, methods=['GET'])

    def index(self):
        return render_template("index.html")

    def matching(self, ctx=2):
        data = flask.request.args
        row, col = int(data['row']), int(data['col'])
        doc1, doc2 = self.coll1[row], self.coll2[col]
        doc1ids, doc2ids = extract_matching_words(doc1, doc2)
        # context doc1
        doc1l = [self.coll1[i] for i in range(max(0, row-ctx), row)]
        doc1r = [self.coll1[i] for i in range(row+1, min(len(self.coll1), row+ctx+1))]
        # context doc2
        doc2l = [self.coll2[i] for i in range(max(0, col-ctx), col)]
        doc2r = [self.coll2[i] for i in range(col+1, min(len(self.coll2), col+ctx+1))]

        return {'doc1': {'left': ' '.join(d.text for d in doc1l),
                         'right': ' '.join(d.text for d in doc1r),
                         'text': doc1.text, 'match': doc1ids,
                         'id': doc1.get_printable_doc_id()},
                'doc2': {'left': ' '.join(d.text for d in doc2l),
                         'right': ' '.join(d.text for d in doc2r),
                         'text': doc2.text, 'match': doc2ids,
                         'id': doc2.get_printable_doc_id()}}

    def heatmap(self):
        rows, cols, vals = scipy.sparse.find(self.sims)
        matches = list(zip(rows.tolist(), cols.tolist(), vals.tolist()))
        min_sim = float(self.sims.min()) if self.min_sim is None else self.min_sim
        max_sim = float(self.sims.max()) if self.max_sim is None else self.max_sim
        data = {'points': [{'row': row,
                            'col': col,
                            'row_id': self.coll1[row].get_printable_doc_id(),
                            'col_id': self.coll2[col].get_printable_doc_id(),
                            'sim': val} for row, col, val in matches],
                'nrow': len(self.coll1),
                'ncol': len(self.coll2),
                'rowName': self.coll1.name,
                'colName': self.coll2.name,
                'meanSim': float(self.sims.data.mean()),
                'maxSim': max_sim,
                'minSim': min_sim}

        return data

    def run(self, debug=True):
        self.app.run(host=self.host, port=self.port, debug=debug)
