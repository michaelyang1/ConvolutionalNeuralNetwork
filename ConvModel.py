from layers import Layers
import numpy as np
import random as rand


class ConvModel:
    act_tensor = list()
    layer_info = list()  # identifies what type of layer (conv, pool, or fc)
    w_tensor = list()
    b_tensor = list()
    dw_tensor = list()
    db_tensor = list()
    dz_tensor = list()  # z = w * x + b
    dact_tensor = list()
    maxpool_masks = list()
    target = list()

    @staticmethod
    def add_layer(layer, done=False):
        """
        Function to add layers to network
        :param layer: (dictionary) attributes of layer
        :param done: (boolean) if finished adding layers
        :return: none
        """
        ConvModel.layer_info.append(layer['Layer operation'])
        if 'Weights and biases' in layer:
            w_b_tup = layer['Weights and biases']
            ConvModel.w_tensor.append(w_b_tup[0])
            ConvModel.b_tensor.append(w_b_tup[1])
        else:
            ConvModel.w_tensor.append(np.array(list()))
            ConvModel.b_tensor.append(np.array(list()))

        if done:
            ConvModel.initialize_arr()

    @staticmethod
    def set_input(input_img):
        if len(input_img.shape) != 3:
            input_img = np.array([input_img])
        ConvModel.act_tensor[0] = input_img

    @staticmethod
    def set_target(target):
        target_arr = np.zeros((len(ConvModel.b_tensor[-1])))
        target_arr[target] = 1.0
        ConvModel.target = target_arr

    @staticmethod
    def initialize_arr():
        ConvModel.act_tensor = [np.array(list())] * (len(ConvModel.layer_info) + 1)
        ConvModel.maxpool_masks = [np.array(list())] * len(ConvModel.layer_info)
        ConvModel.dz_tensor = [np.array(list())] * len(ConvModel.layer_info)
        ConvModel.dact_tensor = [np.array(list())] * len(ConvModel.layer_info)
        ConvModel.dw_tensor = [np.array(list())] * len(ConvModel.layer_info)
        ConvModel.db_tensor = [np.array(list())] * len(ConvModel.layer_info)

    @staticmethod
    def forward_prop():
        for l, l_info in enumerate(ConvModel.layer_info):
            func_name, func_args = l_info[0], l_info[1:]
            act_in = ConvModel.act_tensor[l]
            act_out, dz_out, maxpool_mask = None, None, None

            # perform layer-wise convolution operation
            if func_name == 'convolve3d':
                z_out_3d = Layers.convolve3d(act_in, ConvModel.w_tensor[l], ConvModel.b_tensor[l], func_args[0],
                                             func_args[1])
                dz_out = Layers.drelu(np.array(z_out_3d))
                act_out = Layers.relu(np.array(z_out_3d))
            # perform mlp layer-wise forward pass operation
            if func_name == 'fc_layer':
                act_in = np.ravel(act_in)
                z_out = np.dot(act_in, ConvModel.w_tensor[l]) + ConvModel.b_tensor[l]
                act_out = getattr(Layers, func_args[0])(np.array(z_out))
                dz_out = getattr(Layers, 'd' + func_args[0])(np.array(z_out))
            # perform layer-wise pooling operation
            if func_name == 'maxpool3d':
                pool_outputs = Layers.maxpool3d(input=act_in, filter_dim=func_args[0], stride=func_args[1])
                act_out = pool_outputs[0]
                maxpool_mask = pool_outputs[1]
                dz_out = Layers.drelu(np.array(act_out))

            ConvModel.act_tensor[l + 1] = np.array(act_out)
            ConvModel.dz_tensor[l] = np.array(dz_out)
            ConvModel.maxpool_masks[l] = np.array(maxpool_mask)

    @staticmethod
    def backprop():
        for l in range(len(ConvModel.layer_info) - 1, -1, -1):
            # retrieve dcdact for final layer
            if l == len(ConvModel.layer_info) - 1:
                dact_l = Layers.dact_fc_layer_outer(ConvModel.act_tensor[l + 1], ConvModel.target)
            # retrieve dcdact for layers that link to fc layers
            elif ConvModel.layer_info[l + 1][0] == 'fc_layer':
                dact_l = Layers.dact_fc_layer_inner(ConvModel.w_tensor[l + 1], ConvModel.dz_tensor[l + 1],
                                                    ConvModel.dact_tensor[l + 1]).reshape(ConvModel.dz_tensor[l].shape)
            # retrieve dcdact for layers that link to convolutional layers
            elif ConvModel.layer_info[l + 1][0] == 'convolve3d':
                dact_l = Layers.dact_conv_l(ConvModel.dz_tensor[l], ConvModel.layer_info[l + 1][2],
                                            ConvModel.layer_info[l + 1][1], ConvModel.w_tensor[l + 1],
                                            ConvModel.dz_tensor[l + 1], ConvModel.dact_tensor[l + 1])
            # no backprop operations for layers that link to pooling layers
            else:
                continue

            # retrieve cost derivatives for fully connected layer
            if ConvModel.layer_info[l][0] == 'fc_layer':
                cost_diff = Layers.dcdw_fc(ConvModel.dz_tensor[l], ConvModel.act_tensor[l], dact_l)
            # retrieve cost derivatives for convolutional layer without maxpooling
            elif ConvModel.layer_info[l][0] == 'convolve3d':
                cost_diff = Layers.dcdw_conv_l(ConvModel.act_tensor[l], ConvModel.layer_info[l][2],
                                               ConvModel.layer_info[l][1], ConvModel.w_tensor[l].shape[3],
                                               ConvModel.dz_tensor[l], dact_l)
            # retrieve cost derivatives for convolutional layer with maxpooling
            else:
                dact_l = Layers.apply_maxpool_mask(dact_l, ConvModel.layer_info[l][1], ConvModel.layer_info[l][2],
                                                   ConvModel.maxpool_masks[l])
                cost_diff = Layers.dcdw_conv_l(ConvModel.act_tensor[l - 1], ConvModel.layer_info[l - 1][2],
                                               ConvModel.layer_info[l - 1][1], ConvModel.w_tensor[l - 1].shape[3],
                                               ConvModel.dz_tensor[l - 1], dact_l)

            # store the cost derivatives
            if ConvModel.layer_info[l][0] == 'maxpool3d':
                ConvModel.dw_tensor[l - 1] = cost_diff[0]
                ConvModel.db_tensor[l - 1] = cost_diff[1]
                ConvModel.dact_tensor[l - 1] = cost_diff[2]
            else:
                ConvModel.dw_tensor[l] = cost_diff[0]
                ConvModel.db_tensor[l] = cost_diff[1]
                ConvModel.dact_tensor[l] = cost_diff[2]

    @staticmethod
    def error_calc(data):
        num_correct = 0
        for i in range(len(data)):
            ConvModel.set_target(data[i][0])
            ConvModel.set_input(data[i][1])
            ConvModel.forward_prop()
            if np.argmax(ConvModel.act_tensor[-1]) == data[i][0]:
                num_correct += 1
        return num_correct / len(data)

    @staticmethod
    def train(train_file, test_file, epochs, batch_size, lr):
        """
        Function to train network
        :param train_file: (string) file name of training set
        :param test_file: (string) file name of test set
        :param epochs: (int) number of epochs/iterations to train for
        :param batch_size: (int) size of mini batch
        :param lr: (float) learning rate
        :return: None
        """
        # covert file information into workable data
        data = ConvModel.file_to_data(train_file, test_file)
        print('works')
        train_data, test_data = data[0], data[1]
        dw_update, db_update = 0, 0  # initialize update arrays
        for i in range(epochs):
            rand.shuffle(train_data)  # shuffle training data to induce more noise
            for j in range(len(train_data)):
                # train for one example
                ConvModel.set_target(train_data[j][0])
                ConvModel.set_input(train_data[j][1])
                ConvModel.forward_prop()
                ConvModel.backprop()
                # update the dw and db update arrays
                dw_update += np.array(ConvModel.dw_tensor)
                db_update += np.array(ConvModel.db_tensor)
                # perform mini-batch sgd update
                if j > 0 and j % batch_size == 0 or j == len(train_data) - 1:
                    dw_update /= batch_size
                    db_update /= batch_size
                    ConvModel.w_tensor = ConvModel.w_tensor - lr * dw_update
                    ConvModel.b_tensor = ConvModel.b_tensor - lr * db_update
                    dw_update, db_update = 0, 0  # reset update arrays
                # calculate error percentage at the end of training iteration
                if j == len(train_data) - 1:
                    print('Percent correct (Testing):', ConvModel.error_calc(test_data))

    @staticmethod
    def file_to_data(train_file, test_file):
        train_file = open(train_file, 'r').readlines()[0:1001]
        test_file = open(test_file, 'r').readlines()[0:201]
        train_data, test_data = list(), list()
        for line in train_file[1:]:  # skip header line in file
            # convert all training data to floating values using map()
            train_target = int(line[0])
            train_input = list(map(float, line.split(',')[1:]))
            train_input = np.array(train_input).reshape(28, 28)
            train_input /= 255  # normalize pixel values
            # add to training data array
            train_data.append((train_target, train_input))
        for line in test_file[1:]:
            test_target = int(line[0])
            test_input = list(map(float, line.split(',')[1:]))
            test_input = np.array(test_input).reshape(28, 28)
            test_input /= 255  # normalize pixel values
            # add to test data array
            test_data.append((test_target, test_input))
        return train_data, test_data
