import csv,pprint
from .layers import Base

def get_human_readable(num):
    units=['','K','M','G','T','P']
    idx=0
    while .001*num>1:
        num=.001*num
        idx+=1
    if idx>=len(units):
        return '%.3e'%num
    return '%.3f'%num+units[idx]

def save_csv(blobs,csv_save_path,
             save_items=('name', 'layer_info', 'input', 'out', 'dot', 'add', 'compare','ops', 'weight_size','blob_size'),
             print_detail=True,human_readable=True):
    layers = get_layer_blox_from_blobs(blobs)
    print_list = []
    sum=[0]*len(save_items)
    for layer in layers:
        print_line=[]
        for idx,param in enumerate(save_items):
            item=getattr(layer, param)
            if type(item)==list:
                s=''
                for i in item:
                    s+=' '+str(i)
            else:
                s=str(item)
            try:
                num=int(item)
                sum[idx]+=num
            except:pass
            print_line.append(s)
        print_list.append(print_line)

    if csv_save_path!=None:
        with open(csv_save_path,'w') as file:
            writer=csv.writer(file)
            writer.writerow(save_items)
            for layer in print_list:
                writer.writerow(layer)
    if print_detail:
        sum[0] = 'SUM'
        print_list.append(sum)
        pprint.pprint(print_list,depth=3,width=200)
        print save_items
    else:
        print_list=[]
        for idx,item in enumerate(sum):
            if item>0:
                if human_readable:
                    print_list.append('%s:%s' % (save_items[idx], get_human_readable(item)))
                else:
                    print_list.append('%s:%.3e'%(save_items[idx],item))
        print(print_list)
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

def print_by_blob(blobs,print_items=('name', 'layer_info', 'input', 'out', 'dot', 'add', 'compare','ops', 'weight_size','blob_size')):
    layers=get_layer_blox_from_blobs(blobs)
    print_list = []
    for layer in layers:
        print_list.append([str(getattr(layer, param)) for param in print_items])
    pprint.pprint(print_list, depth=3, width=200)
    return print_list