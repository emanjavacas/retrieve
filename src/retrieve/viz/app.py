
import os

import scipy.sparse
import numpy as np

import flask
from flask import Flask
from flask import render_template


def extract_matching_words(doc1, doc2):
    doc1ws = [w for feat in doc1.get_features() for w in feat.split('--')]
    doc2ws = [w for feat in doc2.get_features() for w in feat.split('--')]
    intersect = set(doc1ws).intersection(doc2ws)
    doc1ids, doc2ids = [], []
    field = 'lemma' if 'lemma' in doc1.fields else 'token'

    for idx, lem in enumerate(doc1.fields[field]):
        if lem in intersect:
            doc1ids.append(idx)
    for idx, lem in enumerate(doc2.fields[field]):
        if lem in intersect:
            doc2ids.append(idx)

    return doc1ids, doc2ids


class VisualizerApp:
    def __init__(self, sims, coll1, coll2=None, 
                 sim_range=(None, None), max_points=5000, sample=False,
                 host='localhost', port=5000):
        """
        Dot-plot visualization app
        """
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
        self.max_points = max_points
        self.sample = sample

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

    def matching(self, ctx=50):
        data = flask.request.args
        row, col = int(data['row']), int(data['col'])
        doc1, doc2 = self.coll1[row], self.coll2[col]
        doc1ids, doc2ids = extract_matching_words(doc1, doc2)

        return {'doc1': {'left': ' '.join(self.coll1.get_doc_context_left(row, ctx)),
                         'right': ' '.join(self.coll1.get_doc_context_right(row, ctx)),
                         'text': doc1.text, 'match': doc1ids,
                         'id': doc1.get_printable_doc_id()},
                'doc2': {'left': ' '.join(self.coll2.get_doc_context_left(col, ctx)),
                         'right': ' '.join(self.coll2.get_doc_context_right(col, ctx)),
                         'text': doc2.text, 'match': doc2ids,
                         'id': doc2.get_printable_doc_id()}}

    def heatmap(self):
        rows, cols, vals = scipy.sparse.find(self.sims)
        n_points, sampled = len(vals), False
        if len(vals) >= self.max_points:
            sampled = True
            if self.sample:
                index = np.random.choice(
                    np.arange(len(vals)), size=self.max_points, replace=False)
            else:
                index = np.argsort(vals)[-self.max_points:]
            rows, cols, vals = rows[index], cols[index], vals[index]

        matches = list(zip(rows.tolist(), cols.tolist(), vals.tolist()))
        min_sim = float(vals.min()) if self.min_sim is None else self.min_sim
        max_sim = float(vals.max()) if self.max_sim is None else self.max_sim
        data = {'points': [{'row': row,
                            'col': col,
                            'row_id': self.coll1[row].get_printable_doc_id(),
                            'col_id': self.coll2[col].get_printable_doc_id(),
                            'sim': val} for row, col, val in matches],
                'nrow': len(self.coll1),
                'ncol': len(self.coll2),
                'rowName': self.coll1.name,
                'colName': self.coll2.name,
                'meanSim': float(vals.mean()),
                'maxSim': max_sim,
                'minSim': min_sim,
                'sampled': sampled,
                'nPoints': n_points}

        return data

    def run(self, debug=True):
        self.app.run(host=self.host, port=self.port, debug=debug)
