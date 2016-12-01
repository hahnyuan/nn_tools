import keras
import proto.net_param as pb
import det1
import numpy as np

def convert_filter(numpy_filter_weight):
    return np.transpose(numpy_filter_weight,(3,2,1,0))

def convert_fc(numpy_fc_weight):
    return np.transpose(numpy_fc_weight,(1,0))

def convert_kmodel(keras_model):
    net=pb.Net()
    weight_name=[i.name for i in keras_model.weights]
    weights=keras_model.get_weights()
    W=[]
    layer=[]
    for name,w in zip(weight_name,weights):
        if 'conv' in name and 'W' in name:
            layer.append(name.split('_W:')[0])
            W.append([convert_filter(w)])
        if 'conv' in name and 'b:' in name:
            W[-1].append(w)
        if 'fc' in name and 'W' in name:
            layer.append(name.split('_W:'[0]))
            W.append([convert_fc(w)])
        if 'fc' in name and 'W' in name:
            W[-1].append(w)
    return W,layer

def save_caffemodel(path,Ws,layer_names):
    net=pb.Net()
    for w,name in zip(Ws,layer_names):
        param=pb.Layer_param(name)
        net.add_layer_with_data(param,w)
    net.save(path)

if __name__=='__main__':
    det1=det1.det1()
    det1.load_weights('det1.kmodel')
    W,layer=convert_kmodel(det1)
    save_caffemodel('../tmp/kdet1.caffemodel',W,layer)
    pass