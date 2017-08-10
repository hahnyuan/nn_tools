# Neural Network Tools: Converter, Constructor and Analyser

 Providing a tool for some fashion neural network frameworks.
 
 The nn_tools is released under the MIT License (refer to the LICENSE file for details).

### features

1. Converting a model between different frameworks.
2. Constructing a model using this tools quickly.
3. Analysing a model, get the operations number(ops) in every layers.

### requirements

- Python2.7
- Each functions in this tools requires corresponding neural network python package (tensorflow pytorch and so on).

# Analyser

## caffe
Before you analyse your network, [Netscope](http://ethereon.github.io/netscope/#/editor)
is recommended to visiualize your network.

Command：`python caffe_analyser.py [-h] prototxt outdir shape`
- The prototxt is the path of the prototxt file.
- The outdir is path to save the csv file.
- The shape is the input shape of the network(split by comma `,`), image shape should be: batch_size, image_height, image_width, channel.

For example `python caffe_analyser.py resnet_18_deploy.prototxt analys_result.csv 1,224,224,3`

# Converter

doc comming soon

# Constructor

doc comming soon
