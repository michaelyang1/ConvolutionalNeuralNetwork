# ConvolutionalNeuralNetwork
This is a convolutional neural network library with various customizable features. This library can create models that can classify black and white or tri-color RGB images with any set of dimensions. 

Prerequisites

Before you continue, ensure you have met the following requirements: 

* You have installed Python 3
* You have installed NumPy 
* You have downloaded the MNIST database 
* You have a basic understanding of convolutional neural networks 

Installation 

1) Install Python with the NumPy library 
2) Download the mnist_train.csv and mnist_test.csv files from https://www.kaggle.com/oddrationale/mnist-in-csv 
3) Place both files in the project folder 

Usage 

* To build the network, call the add_layer function from the ConvModel class 
* For the add_layer parameter, supply what type of layer you want via the Layer class as the argument
* The 3 types of layers are : convolutional layers, maxpooling layers, and fully connected layers
* In the arguments for the final layer, supply the keyword argument done=True as the second argument for the add_layer function, indicating the completion of creating the model architecture 
* To train the network, call the train function from the ConvModel class 
* Note you must first create the network via the add_layer function before you can begin training
* Layers are added sequentially, meaning the final layer added is also the output layer of the network 

Contact Information 

* Email : myang1394@gmail.com


