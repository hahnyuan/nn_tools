import csv,pprint
from .layers import Base

def save_csv(blobs,csv_save_path,save_items=('name', 'layer_info', 'input', 'out', 'dot', 'add', 'compare','flops', 'weight_size','blob_size')):
    layers = get_layer_blox_from_blobs(blobs)
    print_list = []
    for layer in layers:
        print_list.append([str(getattr(layer, param)) for param in save_items])
    if csv_save_path!=None:
        with open(csv_save_path,'w') as file:
            writer=csv.writer(file)
            writer.writerow(save_items)
            for layer in print_list:
                writer.writerow(layer)
    pprint.pprint(print_list,depth=3,width=200)
    print 'saved!'

def get_layer_blox_from_blobs(blobs):
    layers=[]
    def creator_search(blob):
        for father in blob.father:
            if isinstance(father,Base) and father not in layers:
                layers.append(father)
                if father.muti_input==True:
                    for input in father.input:
                        creator_search(input)
                else:
                    creator_search(father.input)
    for blob in blobs:
        creator_search(blob)
    return layers

def print_by_blob(blobs,print_items=('name', 'layer_info', 'input', 'out', 'dot', 'add', 'compare','flops', 'weight_size','blob_size')):
    layers=get_layer_blox_from_blobs(blobs)
    print_list = []
    for layer in layers:
        print_list.append([str(getattr(layer, param)) for param in print_items])
    pprint.pprint(print_list, depth=3, width=200)
    return print_list