# A compact network learning model for distribution regression

## Distribution regression network (DRN)

Despite the superior performance of deep learning in many applications, challenges remain in the area of regression on function spaces. In particular, neural networks are unable to encode function inputs compactly as each node encodes just a real value. We propose a novel idea to address this shortcoming: to encode an entire function in a single network node. To that end, we design a compact network representation that encodes and propagates functions in single nodes for the distribution regression task. Our proposed distribution regression network (DRN) achieves higher prediction accuracies while using fewer parameters than traditional neural networks.

Kou, Connie Khor Li, Hwee Kuan Lee, and Teck Khim Ng. "A compact network learning model for distribution regression." Neural Networks 110 (2019): 199-212.

https://doi.org/10.1016/j.neunet.2018.12.007

Video: https://sites.google.com/view/conniekou/research/distribution-regression-network

## Source code
This source code trains DRN for the Ornstein-Uhlenbeck dataset. 

## Pre-requisites
Tensorflow and Numpy
