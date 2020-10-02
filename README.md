# ECNN
This package uses adaptive evolutionary algorithms in order to train, evaluate, and evolve Convolutional Neural Networks. It includes a simple interface with TensorFlow and allows for parallel model training across multiple GPUs.

## Getting Started

The provided code uses the CIFAR 10 dataset as input. The can be downloaded [here](https://www.cs.toronto.edu/~kriz/cifar.html). The dataset should be saved in the 'datasets/cifar10' directory. Other datasets can be adapted to use the DataSet class provided in dataset.py.

Tournament defautls can be set in defaults.py.

### Prerequisities
* Python 3
* Numpy
* TensorFlow
* namedlist
* Pickle

### Examples 
For full list of results please see the [Report](https://github.com/alazareva/ECNN/blob/master/report.pdf).  Example tournament results showing mutations applied at each generation and resulting CNN performance.

![Ex1](https://github.com/alazareva/ECNN/blob/master/examples/screenshot_2.jpg)
![Ex2](https://github.com/alazareva/ECNN/blob/master/examples/screenshot_3.jpg)

## Authors

* **Anastasiya Lazareva** - *Initial work* - [alazareva](https://github.com/alazareva)

