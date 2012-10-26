#Accessors for matrix, for transparent work with matrix minors and transposed matrices
class ColumnView(object):
    """Reaw-Write view of a matrix column"""
    def __init__(self, m, idx):
        self.m = m
        self.idx = idx

    def __len__(self):
        return len(self.m)

    def __getitem__(self, i):
        if isinstance(i, slice):
            idx = self.idx
            return [mi[idx] for mi in self.m[i]]
        else:
            return self.m[i][self.idx]

    def __setitem__(self, i, v):
        if isinstance(i, slice):
            idx = self.idx
            for (mi, vi) in zip(self.m[i], v):
                mi[idx] = vi
        else:
            self.m[i][self.idx] = v

    def __str__(self):
        return str(list(self))

    def __repr__(self):
        return repr(list(self))

    def __iter__(self):
        i = self.idx
        for row in self.m:
            yield row[i]

class TransposedView(object):
    """Read-Write view of a transposed matrix"""
    def __init__(self, m):
        n = len(m[0])
        self.columns = [ColumnView(m, i) for i in xrange(n)]
    def __len__(self):
        return len(self.columns)
    def __getitem__(self, i):
        return self.columns[i]
    def __setitem__(self, i, v):
        if isinstance(i, slice):
            for col, vi in zip(self.columns[i], v):
                col[:] = vi
        else:
            self.columns[i][:] = v
    def __str__(self):
        return str(self.columns)
    def __repr__(self):
        return repr(self.columns)
    def __iter__(self):
        return self.columns.__iter__()
    def original(self):
        return m
