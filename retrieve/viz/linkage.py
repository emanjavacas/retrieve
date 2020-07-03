
import json

import scipy.sparse


class Linkage:
    def __init__(self, coll1, coll2, sims, visualizer):
        if len(coll1) > len(coll2):
            self.rowname = coll2.name
            self.colname = coll1.name
            sims = sims.T
        else:
            self.rowname = coll1.name
            self.colname = coll2.name
        self.colls = {coll1.name: coll1, coll2.name: coll2}
        self.sims = sims
        self.visualizer = visualizer

    def export_sims(self):
        output = {}
        rows, cols, sims = scipy.sparse.find(self.sims)
        output['ncol'] = len(self.colls[self.coll1name])
        output['nrow'] = len(self.colls[self.coll2name])
        output['n_matches'] = len(rows)
        output['points'] = []
        for row, col, sim in zip(rows.tolist(), cols.tolist(), sims.tolist()):
            output['points'].append({'col': col, 'row': row, 'sim': sim})
        output['rowName'] = self.rowname
        output['colName'] = self.colname

        # need to get similarity domain from sims to compute colors
        return output

    def get_document_data(self, collname, doc_idx):
        doc = self.colls[collname][doc_id]
        # 


# from retrieve.data import Collection
# from retrieve.corpora import load_vulgate
# old, new = load_vulgate(path='data/texts/blb.lxx.csv', split_testaments=True)
# old[12]
