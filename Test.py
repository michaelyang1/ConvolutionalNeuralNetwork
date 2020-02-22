from convolutionalneuralnetwork import ConvModel
from layers import Layers

if __name__ == '__main__':
    # create Le-Net 5 ConvNet architecture by adding layers via Layer class
    ConvModel.add_layer(Layers.conv_layer(20, 5, padding='valid', input_shape=(28, 28, 1)))
    ConvModel.add_layer(Layers.maxpooling(2, stride=2))
    ConvModel.add_layer(Layers.conv_layer(40, 5, padding='valid'))
    ConvModel.add_layer(Layers.maxpooling(2, stride=2))
    ConvModel.add_layer(Layers.fc_layer(100, activation='relu'))
    ConvModel.add_layer(Layers.fc_layer(10, activation='softmax'), done=True)
    # train the network on MNIST database with 1000 training epochs, mini-batch size of 10, and learning rate of 0.1
    ConvModel.train('mnist_train.csv', 'mnist_test.csv', 1000, 10, 0.1)
