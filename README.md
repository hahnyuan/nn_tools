# Neural Network Tools: Converter, Constructor and Analyser

 Providing a tool for some fashion neural network frameworks.
 
 The nn_tools is released under the MIT License (refer to the LICENSE file for details).

### features

1. Converting a model between different frameworks.
2. Some convenient tools of manipulate caffemodel and prototxt quickly, see [Caffe model](https://github.com/hahnyuan/nn_tools/tree/master/Caffe).
3. Analysing a model, get the operations number(ops) in every layers.

### requirements

- Python2.7
- Each functions in this tools requires corresponding neural network python package (tensorflow pytorch and so on).

# Analyser

The analyser can analyse all the model layers' [input_size, output_size, multiplication ops, addition ops, comparation ops, tot ops, weight size and so on] given a input tensor size, which is convenint for model deploy analyse.

## caffe
Before you analyse your network, [Netscope](http://ethereon.github.io/netscope/#/editor)
is recommended to visiualize your network.

Command：`python caffe_analyser.py [-h] prototxt outdir shape`
- The prototxt is the path of the prototxt file.
- The outdir is path to save the csv file.
- The shape is the input shape of the network(split by comma `,`), in caffe image shape should be: batch_size, image_height, image_width, channel.

For example `python caffe_analyser.py resnet_18_deploy.prototxt analys_result.csv 1,224,224,3`

## Pytorch
Supporting analyse the inheritors of torch.nn.Moudule class.

Command：`pytorch_analyser.py [-h] [--out OUT] [--class_args ARGS] path class_name shape`
- The path is the python file path which contaning your class.
- The class_name is the class name in your python file.
- The shape is the input shape of the network(split by comma `,`), in pytorch image shape should be: batch_size, channel, image_height, image_width.
- The out (optinal) is path to save the csv file, default is '/tmp/pytorch_analyse.csv'.
- The class_args (optional) is the args to init the class in python file, default is empty.

For example `python pytorch_analyser.py tmp/pytorch_analysis_test.py ResNet218 1,3,224,224`

# Converter

doc comming soon

