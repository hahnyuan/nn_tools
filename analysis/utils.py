import csv,pprint
from .layers import Base

DEFAULT_ITEMS=('name', 'layer_info', 'input', 'out', 'dot', 'add', 'compare', 'ops', 'weight_size', 'activation_size')

def get_human_readable(num):
    units=['','K','M','G','T','P']
    idx=0
    while .001*num>1:
        num=.001*num
        idx+=1
    if idx>=len(units):
        return '%.3e'%num
    return '%.3f'%num+units[idx]

def get_items_data(item_names, layers):
    items = []
    layers_sum = [0] * len(item_names)
    for layer in layers:
        print_line = []
        for idx, param in enumerate(item_names):
            item = getattr(layer, param)
            if type(item) == list:
                s = ''
                for i in item:
                    s += ' ' + str(i)
            else:
                s = str(item)
            try:
                num = int(item)
                layers_sum[idx] += num
            except:
                pass
            print_line.append(s)
        items.append(print_line)
    return items,layers_sum

def save_csv(layers, csv_save_path='/tmp/analyse.csv',
             save_items=DEFAULT_ITEMS,
             print_detail=False, human_readable=True):
    # layers = get_layer_blox_from_blobs(blobs)
    items,layers_sum=get_items_data(save_items,layers)
    if csv_save_path!=None:
        with open(csv_save_path,'w') as file:
            writer=csv.writer(file)
            writer.writerow(save_items)
            for layer in items:
                writer.writerow(layer)
        print('saved at {}!'.format(csv_save_path))

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

def print_table(datas,names):

    types=[]
    for i in datas[0]:
        try:
            i=int(float(i))
            types.append('I')
        except:
            types.append('S')
    for l in datas:
        s=''
        for i,t in zip(l,types):
            if t=='I':

                i=int(float(i))
                s+=('%.1E'%i).center(10)
            else:
                i=str(i)
                if len(i)>20:
                    i=i[:17]+'...'
                s+=i.center(20)
            s+='|'
        print(s)
    s = ''
    for i,t in zip(names,types):

        if t == 'I':
            s += i.center(10)
        else:
            if len(i) > 20:
                i = i[:17] + '...'
            s += i.center(20)
        s += '|'
    print(s)

def print_by_layers(layers,print_detail=True,human_readable=True, print_items=DEFAULT_ITEMS):
    items, layers_sum=get_items_data(print_items,layers)
    if print_detail:
        layers_sum[0] = 'SUM'
        items.append(layers_sum)
        print_table(items,print_items)
    else:
        items=[]
        for idx,item in enumerate(layers_sum):
            if item>0:
                if human_readable:
                    items.append('%s:%s' % (print_items[idx], get_human_readable(item)))
                else:
                    items.append('%s:%.3e'%(print_items[idx],item))
        print(items)
    return items

def print_by_blob(blobs, print_items=DEFAULT_ITEMS):
    layers=get_layer_blox_from_blobs(blobs)
    print_by_layers(layers,print_items)