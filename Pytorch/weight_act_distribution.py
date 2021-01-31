import json

import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet,mobilenet,mnasnet
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

batch_size=64

def get_val_loader():
    train_path = '/datasets/imagenet/val'
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    imagenet_data = torchvision.datasets.ImageFolder(train_path, transform=transform)
    data_loader = torch.utils.data.DataLoader(
        imagenet_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    return data_loader


loader=get_val_loader()

save_weight=0

models=[resnet.resnet18,resnet.resnet34]
models=[mobilenet.mobilenet_v2]
# models=[mnasnet.mnasnet0_5]
for model_fun in models:
    
    model=model_fun(True)
    model.eval()
    model.cuda()
    model_name=model.__class__.__name__
    print(model_name)

    weights=[]
    names=[]
    w_dict={'layer':[],'values':[]}
    
    if save_weight:
        for name,m in model.named_modules():
            if isinstance(m,nn.Conv2d):
                w=m.weight.detach().cpu().numpy()
                w_log=np.log2(w).reshape(-1)
                weights.append(w_log)
                w_dict['layer']+=[name]*len(w_log)
                w_dict['values']+=[w_log]
                names.append(name)
        w_dict['values']=np.concatenate(w_dict['values'])
        plt.figure(figsize=[20,10])
        sns.violinplot(x='layer',y='values',data=w_dict,scale='width')
        plt.title(f"{model_name} weight (log2)")
        plt.xticks(rotation=45)
        plt.savefig(f'{model_name}_weight_layerwise.jpg')
        plt.clf()
    # w_log=np.concatenate(weights)
    # sns.histplot(w_log)
    # plt.title(model_name+"_weight(log2)")
    # plt.savefig(f'{model_name}.jpg')
    # plt.clf()

    i_logs=[]
    act={'layer':[],'value':[],'pos':[]}
    grad={'layer':[],'value':[],'pos':[]}
    layer_names={}

    bn_o_logs=[]
    def bn_hook(m,i,o):
        layer=layer_names[m]
        i=i[0].detach().view(-1)
        idx=i.nonzero()
        i=i[idx].abs().sort()[0][64:-64:batch_size].cpu().numpy()
        i_log=np.nan_to_num(np.log2(i).reshape(-1))
        act['layer']+=[layer]*len(i_log)
        act['value'].extend(i_log)
        act['pos']+=['BN input']*len(i_log)

        i=o.detach().view(-1)
        idx=i.nonzero()
        i=i[idx].abs().sort()[0][64:-64:batch_size].cpu().numpy()
        i_log=np.nan_to_num(np.log2(i).reshape(-1))
        act['layer']+=[layer]*len(i_log)
        act['value'].extend(i_log)
        act['pos']+=['BN output']*len(i_log)

    def conv_back_hook(m,gi,go):
        layer=layer_names[m]
        i=go[0].detach().view(-1)
        idx=i.nonzero()
        i=i[idx].abs().sort()[0][64:-64:batch_size].cpu().numpy()
        i_log=np.nan_to_num(np.log2(i).reshape(-1))
        act['layer']+=[layer]*len(i_log)
        act['value'].extend(i_log)
        act['pos']+=['Weight grad']*len(i_log)

        i=go[1].detach().view(-1)
        idx=i.nonzero()
        i=i[idx].abs().sort()[0][64:-64:batch_size].cpu().numpy()
        i_log=np.nan_to_num(np.log2(i).reshape(-1))
        act['layer']+=[layer]*len(i_log)
        act['value'].extend(i_log)
        act['pos']+=['Weight grad']*len(i_log)
    
    for name,m in model.named_modules():
        layer_names[m]=name
        if isinstance(m,nn.BatchNorm2d):
            m.register_forward_hook(bn_hook)
        if isinstance(m,nn.Conv2d):
            m.register_backward_hook(conv_back_hook)

    for d,t in loader:
        print(t)
        d=d.cuda()
        model(d)
        break
    
    plt.figure(figsize=[20,10])
    sns.violinplot(x='layer',y='value',hue='pos',data=act,scale='width',split=True)
    plt.title(f"{model_name} activations(log2)")
    plt.xticks(rotation=45)
    plt.savefig(f'{model_name}_activations_layerwise.jpg')
    plt.clf()
    