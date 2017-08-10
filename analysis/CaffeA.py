from collections import OrderedDict
from layers import *
from roi import *

def profilling(net,input=None):
    # input is either a Blob with the shape of (batch,h,w,c) or a dict of them
    if isinstance(input,dict):
        blob_dict = OrderedDict(input)
        not_ref = [input[k] for k in input]
    else:
        blob_dict = OrderedDict({'data': input})
        not_ref=[input]
    for i, layer in enumerate(net.net.layer):
        out = None
        if len(layer.top) == 1 and len(layer.bottom) == 1:
            # first remove the node from not_ref
            try:not_ref.remove(blob_dict[layer.bottom[0]])
            except:pass
            if layer.type == 'Convolution':
                param = layer.convolution_param
                out = conv(blob_dict[layer.bottom[0]], param.kernel_size, param.num_output, param.stride,
                             param.pad, None, layer.name)
            if layer.type == 'InnerProduct':
                param=layer.inner_product_param
                out= fc(blob_dict[layer.bottom[0]],param.num_output,None,layer.name)
            if layer.type == 'ReLU':
                out = Activation(blob_dict[layer.bottom[0]], 'relu', layer.name)
            if layer.type == 'Pooling':
                param = layer.pooling_param
                out = Pool(blob_dict[layer.bottom[0]], param.kernel_size, param.stride,
                             param.pad, layer.name,param.pool,ceil=True)
            if layer.type == 'Normalize':
                out = Norm(blob_dict[layer.bottom[0]], 'norm', layer.name)
            if layer.type == 'BatchNorm':
                out= Norm(blob_dict[layer.bottom[0]],'batch_norm',layer.name)
            if layer.type== 'LRN':
                out= Norm(blob_dict[layer.bottom[0]],'lrn',layer.name)
            if layer.type == 'Permute':
                shape=[blob_dict[layer.bottom[0]][dim-1] for dim in layer.permute_param.order[1:]]
                out = Permute(blob_dict[layer.bottom[0]],shape,layer.name)
            if layer.type == 'Flatten':
                out = Flatten(blob_dict[layer.bottom[0]], layer.name)
            if layer.type == 'Scale':
                out =Scale (blob_dict[layer.bottom[0]], name = layer.name)
            if out:
                blob_dict[layer.top[0]] = out()
                not_ref.append(blob_dict[layer.top[0]])
            else:
                assert 'layer type: %s cannot be P' % (layer.type)
        elif len(layer.bottom)>1:
            # for multi input layer
            for bottom in layer.bottom:
                try:not_ref.remove(blob_dict[bottom])
                except:pass
            if layer.type=='Eltwise':
                param=layer.eltwise_param
                out = Eltwise([blob_dict[bottom] for bottom in layer.bottom],
                              type=param.EltwiseOp.Name(param.operation),name=layer.name)
            if layer.type=='PSROIPooling':
                param=layer.psroi_pooling_param
                out = PSROIPool(blob_dict[layer.bottom[0]],blob_dict[layer.bottom[1]],
                                param.output_dim,param.group_size)
            if layer.type=='ROIPooling':
                param=layer.roi_pooling_param
                out = ROIPool(blob_dict[layer.bottom[0]],blob_dict[layer.bottom[1]],
                              param.pooled_w,param.pooled_h,layer.name)
            if layer.type == "Concat":
                out = Concat([blob_dict[bottom] for bottom in layer.bottom],layer.name)
            if out:
                blob_dict[layer.top[0]] = out()
                not_ref.append(blob_dict[layer.top[0]])
            else:
                assert 'layer type: %s cannot be P' % (layer.type)
    return blob_dict,not_ref