# Neural Network Tools: Converter, Constructor and Analyser

 Providing a tool for some fashion neural network frameworks.
 
 The nn_tools is released under the MIT License (refer to the LICENSE file for details).


- [Converter](#Converter)
  - [Pytorch->Caffe](#Pytorch-to-Caffe)
- [Analyser](#Analyser)
  - [Caffe Analyser](##Caffe-Analyser)
  - [Pytorch Analyser](##Pytorch-Analyser)
  - [Mxnet Analyser](##Mxnet-Analyser)
- [Some useful functions](#Some-useful-functions)

### features

1. Converting a model between different frameworks.
2. Some convenient tools of manipulate caffemodel and prototxt quickly(like get or set weights of layers), 
see [nn_tools.Caffe](https://github.com/hahnyuan/nn_tools/tree/master/Caffe).
3. Analysing a model, get the operations number(ops) in every layers.

### requirements

- Python2.7 or Python3.x
- Each functions in this tools requires corresponding neural network python package (tensorflow pytorch and so on).

# Converter

## Pytorch to Caffe

The new version of pytorch_to_caffe supporting the newest version(from 0.2.0 to 1.0) of pytorch.
NOTICE: The transfer output will be somewhat different with the original model, caused by implementation difference.

- Supporting layers types: conv2d, transpose_conv2d, linear, max_pool2d, avg_pool2d, dropout,
 relu, prelu, threshold(only value=0),softmax, batch_norm, instance_norm

- Supporting operations: torch.split, torch.cat
- Supporting tensor Variable operations: var.view, var.mean, var.sum, var.contiguous, + (add), += (iadd), -(sub), -=(isub)
 \* (mul) *= (imul) / (div)
- The not supporting operations will transferred to a Python layer in Caffe. You can implemented it by yourself.
- Testify whether your transformed Caffe model is workable. See `tmp/testify_pytorch_to_caffe.py`.

The supported above can transfer many kinds of nets.
The tested network:
- AlexNet(tested)
- VGG(tested)
- ResNet(tested)
- Inception_V3(tested)
- SqueezeNet(tested)

The supported layers concluded the most popular layers and operations.
 The other layer types will be added soon, you can ask me to add them in issues.

Note: You need `net.eval()` before converting the pytorch networks.

Example: please see file `example/<alexnet/resnet/inception_v3>_pytorch_to_caffe.py`.

```
$python3 example/alexnet_pytorch_to_caffe.py

Add blob         blob0         : torch.Size([1, 3, 226, 226])
Processing Layer: features.0
Add blob       conv_blob1      : torch.Size([1, 64, 55, 55])
Processing Layer: features.1
 ...
Transform Completed
```

If you have compiled Pycaffe and set the pycaffe path.
You can run testify_pytorch_to_caffe to test whether the output of
every Caffe layer is the same as the output in Pytorch.

```
$python3 example/testify_pytorch_to_caffe_example.py
TEST layer features_0: PASS
TEST layer features_1: PASS
...
TEST output
TEST output: PASS
```


# Analyser

The analyser can analyse all the model layers' [input_size, output_size, multiplication ops, addition ops, 
comparation ops, tot ops, weight size and so on] given a input tensor size, which is convenint for model deploy analyse.

## Caffe Analyser
Before you analyse your network, [Netscope](http://ethereon.github.io/netscope/#/editor)
is recommended to visiualize your network.

Command：`python caffe_analyser.py [-h] prototxt outdir shape`
- The prototxt is the path of the prototxt file.
- The outdir is path to save the csv file.
- The shape is the input shape of the network(split by comma `,`), in caffe image shape should be: 
batch_size, channel, image_height, image_width.

For example `python caffe_analyser.py resnet_18_deploy.prototxt analys_result.csv 1,3,224,224`

## Pytorch Analyser
Supporting analyse the inheritors of torch.nn.Moudule class.

Command：`pytorch_analyser.py [-h] [--out OUT] [--class_args ARGS] path name shape`
- The path is the python file path which contaning your class.
- The name is the class name or instance name in your python file.
- The shape is the input shape of the network(split by comma `,`), in pytorch image shape should be:
batch_size, channel, image_height, image_width.
- The out (optinal) is path to save the csv file, default is '/tmp/pytorch_analyse.csv'.
- The class_args (optional) is the args to init the class in python file, default is empty.

For example `python pytorch_analyser.py example/resnet_pytorch_analysis_example.py resnet18 1,3,224,224`


## Mxnet Analyser
Supporting analyse the inheritors of mxnet.sym.

Command：`mxnet_analyser.py [-h] [--out OUT] [--func_args ARGS] [--func_kwargs FUNC_KWARGS] path name shape`
- The path is the python file path which contaning your symbol definition.
- the symbol object name or function that generate the symbol in your python file.
- The shape is the input shape of the network(split by comma `,`), in mxnet image shape should be:
batch_size, channel, image_height, image_width.
- The out (optinal) is path to save the csv file, default is '/tmp/mxnet_analyse.csv'.
- The func_args (optional) is the args to init the class in python file, default is empty.

For example `python mxnet_analyser.py example/mobilenet_mxnet_symbol.py get_symbol 1,3,224,224`


# Some useful functions

## funcs.py

- **get_iou(box_a, box_b)** intersection over union of two boxes
- **nms(bboxs,scores,thresh)** Non-maximum suppression
- **Logger** print some str to a file and stdout with H M S

