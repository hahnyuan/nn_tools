import lmdb
import os

class LMDB(object):
    def __init__(self,lmdb_dir):
        self.env=lmdb.Environment(lmdb_dir)


if __name__=='__main__':
    pass