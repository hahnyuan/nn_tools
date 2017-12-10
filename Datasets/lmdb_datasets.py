import caffe_lmdb
from . import lmdb_data_pb2 as pb2
import numpy as np
import multiprocessing
import os


class LMDB(object):
    def __init__(self,lmdb_dir):
        self.env=caffe_lmdb.Environment(lmdb_dir, map_size=int(1e12))

    # --------------------------------------
    # for LMDB writer



    # --------------------------------------
    # for LMDB reader
class LMDB_generator(object):
    def __init__(self,lmdb_dir):
        self.env=caffe_lmdb.Environment(lmdb_dir, map_size=int(1e12))

    def generate_datum(self,data,target,other=None):
        datum = pb2.Datum()
        datum.data=data
        datum.target=target
        if other:
            datum.other.extend(other)
        # data=np.fromstring(datum.data,np.uint8).reshape(datum.other[0],np.uint16)
        return datum

    def generate_dataset(self,datas,targets,others=None):
        dataset = pb2.Dataset()
        assert len(datas)==len(targets),ValueError('the lengths of datas and targets are not the same')
        for idx in xrange(len(datas)):
            try:
                if others==None:
                    datum=self.generate_datum(datas[idx],targets[idx])
                else:
                    datum = self.generate_datum(datas[idx], targets[idx],others[idx])
            except:
                print('generate the datum failed at %d, continue it'%idx)
                continue
            dataset.datums.extend([datum])
        return dataset

    def commit_dataset(self,dataset,idx):
        txn=self.env.begin(write=True)
        txn.put(str(idx),dataset.SerializeToString())
        txn.commit()

    def write_classification_lmdb(self, data_loader, num_per_dataset=3000, write_shape=False):
        # torch_data_loader are iterator that iterates a (data,target)
        # data should be a numpy array
        # target should be a int number
        datas=[]
        targets=[]
        others=[]
        for idx,(data,target) in enumerate(data_loader):
            datas.append(data.tobytes())
            targets.append(bytes(target))
            if write_shape:
                others.append([np.array(data.shape, np.uint16).tobytes()])
            if (idx%num_per_dataset==0 and idx!=0) or (idx==len(data_loader)-1):
                print('lmdb write at image %d'%idx)
                dataset=self.generate_dataset(datas,targets,write_shape and others or None)
                self.commit_dataset(dataset,np.ceil(1.*idx/num_per_dataset))
                datas=[]
                targets=[]
                others = []

    def write_lmdb_mutiprocess(self,torch_data_loader,num_thread=10,num_per_dataset=6000):
        # TODO mutiprocess write lmdb
        pass


if __name__=='__main__':
    pass