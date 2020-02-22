import numpy as np
import warnings


class Layers:
    feature_shape = list()

    @staticmethod
    def create_filter(filter_dim, input_shape):
        """
        Function to create filters with Xavier initialization
        :param filter_dim: (int) filter dimensions
        :param input_shape: (array-like) shape of input feature maps
        :return: (2d array) single sheet of individual filter
        """
        fan_in = input_shape[0] * input_shape[1]
        return np.random.randn(filter_dim, filter_dim) * np.sqrt(2.0 / fan_in)

    @staticmethod
    def create_weights(fan_in, fan_out):
        """
        Function to create weights in fully connected layers with Xavier initialization
        :param fan_in: (int) number of neurons in layer
        :param fan_out: (int) number of neurons in next layer
        :return: (2d array) initialized weights for layer
        """
        return np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)

    @staticmethod
    def create_biases(fan_in, bias):
        """
        Function to create biases in fully connected layers
        :param fan_in: (int) number of neurons in layer
        :param bias: (int) value of bias
        :return: (1d array) initialized bias array for entire layer
        """
        return np.full(fan_in, bias)

    @staticmethod
    def conv_layer(num_filters, filter_dim, stride=1, padding='valid', bias=0, activation='relu', input_shape=None):
        """
        Function to create convolutional layer
        :param num_filters: (int) number of filters in layer
        :param filter_dim: (int) filter dimension
        :param stride: (int) convolution stride
        :param padding: (string) type of padding
        :param bias: (int) value of bias for layer
        :param activation: (string) activation function
        :param input_shape: (tuple) shape of input feature map
        :return: (dictionary) information about the layer
        """
        # same padding only works for odd filter sizes
        if filter_dim % 2 == 0 and padding == 'same':
            warnings.warn('Even filter dimensions.')

        # record initial input shape
        if input_shape:
            Layers.feature_shape = list(input_shape)

        # create filters and biases for layer with given dimensions
        filters_layer, biases_layer = list(), list()
        for i in range(num_filters):
            filter = list()  # one individual filter
            for j in range(Layers.feature_shape[2]):
                filter.append(Layers.create_filter(filter_dim, Layers.feature_shape))
            filters_layer.append(np.array(filter))
            biases_layer.append(bias)

        # determine the shape of next layer
        if padding == 'valid':
            new_2d_shape = ((Layers.feature_shape[0] - filter_dim) // stride + 1,
                            (Layers.feature_shape[1] - filter_dim) // stride + 1)
            Layers.feature_shape[0], Layers.feature_shape[1] = new_2d_shape[0], new_2d_shape[1]
        Layers.feature_shape[2] = num_filters

        # return a dictionary containing all necessary information for the layer
        layer_info = {'Weights and biases': (np.array(filters_layer), np.array(biases_layer))}
        layer_info.update({'Layer operation': ('convolve3d', stride, padding, activation)})
        return layer_info

    @staticmethod
    def fc_layer(neurons, activation='relu', bias=0):
        """
        Function to create fully connected layer
        :param neurons: (int) number of neurons in layer
        :param activation: (string) activation function
        :param bias: (int) value of bias for layer
        :return: (dictionary) information describing layer
        """
        # flatten the dimensions of conv layer preceding fc layer
        if isinstance(Layers.feature_shape, list):
            Layers.feature_shape = Layers.feature_shape[0] * Layers.feature_shape[1] * Layers.feature_shape[2]

        # create weights and biases for layer
        weights_layer = Layers.create_weights(Layers.feature_shape, neurons)
        biases_layer = Layers.create_biases(neurons, bias)
        Layers.feature_shape = neurons

        # return a dictionary containing all necessary information for the layer
        layer_info = {'Weights and biases': (np.array(weights_layer), np.array(biases_layer))}
        layer_info.update({'Layer operation': ('fc_layer', activation)})
        return layer_info

    @staticmethod
    def maxpooling(filter_dim, stride):
        """
        Function to create maxpooling layer
        :param filter_dim: (int) filter dimension
        :param stride: (int) stride
        :return: (dictionary) information describing layer
        """
        # determine shape of pooled 2d feature map
        new_2d_shape = ((Layers.feature_shape[0] - filter_dim) // stride + 1,
                        (Layers.feature_shape[1] - filter_dim) // stride + 1)

        # obtain new shape (the number of channels doesn't change during maxpooling)
        Layers.feature_shape[0], Layers.feature_shape[1] = new_2d_shape[0], new_2d_shape[1]

        # return a dictionary containing all necessary information for the 'layer'
        layer_info = {'Layer operation': ('maxpool3d', filter_dim, stride)}
        return layer_info

    @staticmethod
    def relu(w_sums):
        """
        Rectified Linear Unit function
        :param w_sums: (array-like) weighted sum array
        :return: (array-like) activation array after relu
        """
        w_sums[w_sums < 0] = 0
        return w_sums

    @staticmethod
    def drelu(w_sums):
        """
        Derivative of the Rectified Linear Unit function
        :param w_sums: (array-like) weighted sum array
        :return: (array-like) derivative of relu activation wrt w_sums
        """
        w_sums[w_sums > 0] = 1
        w_sums[w_sums <= 0] = 0
        return w_sums

    @staticmethod
    def softmax(w_sums):
        """
        Softmax function used in output layer
        :param w_sums: (array-like) weighted sum array
        :return: (array-like) activation array after softmax
        """
        t = np.power(np.e, w_sums)
        output = t / np.sum(t)
        return output

    @staticmethod
    def dsoftmax(w_sums):
        """
        Derivative of softmax function
        :param w_sums: (array-like) weighted sum array
        :return: (array-like) derivative of softmax activation wrt w_sums
        """
        t = np.power(np.e, w_sums)
        t_sum = np.sum(t)
        output = (t_sum * t - t ** 2) / (t_sum ** 2)
        return output

    @staticmethod
    def maxpool3d(input, filter_dim, stride):
        """
        Function to perform max pooling on input
        :param input: (3d tensor) input feature map
        :param filter_dim: (int) filter dimensions
        :param stride: (int) stride
        :return: (3d tensor) pooled input
        """

        # determine output shape
        pooled_output_shape = (input.shape[0], (input.shape[1] - filter_dim) // stride + 1,
                               (input.shape[2] - filter_dim) // stride + 1)
        # initialize pool variables
        pooled_input = list()
        pooled_masks = np.zeros(input.shape)
        pooled_indicies = np.arange(input.shape[0])

        # perform max pooling across input space
        for h in range(0, input.shape[1], stride):
            for w in range(0, input.shape[2], stride):
                region1 = input[:, h:h + filter_dim, w:w + filter_dim]
                region2 = pooled_masks[:, h:h + filter_dim, w:w + filter_dim]
                if region1.shape[1:] == (filter_dim, filter_dim):
                    pool_max = np.amax(region1, axis=(1, 2))
                    pooled_input.append(pool_max)
                    loc = region1.reshape(region1.shape[0], -1).argmax(-1)
                    loc = np.array(np.unravel_index(loc, region1.shape[1:]))
                    region2[pooled_indicies, loc[0], loc[1]] = 1
                else:
                    break

        # reshape pooled output
        pooled_input = np.transpose(pooled_input).reshape(pooled_output_shape)
        return pooled_input, pooled_masks

    @staticmethod
    def convolve3d(input, filter, bias, stride=1, padding='valid'):
        """
        Function to perform convolution/cross-correlation on 3d input
        :param input: (3d tensor) input feature map (activations in layer)
        :param filter: (4d tensor) filters in layer
        :param stride: (int) stride
        :param padding: (string) type of padding
        :param bias: (1d array) biases in layer
        :return: (3d tensor) convolution output for next layer
        """
        # determine output shape
        filter_dim = filter.shape[2]
        if padding == 'valid':
            conv_output_shape = (filter.shape[0], (input.shape[1] - filter_dim) // stride + 1,
                                 (input.shape[2] - filter_dim) // stride + 1)
        else:
            conv_output_shape = (filter.shape[0], input.shape[1], input.shape[2])

        # apply padding if necessary
        input = Layers.apply_padding(input, padding, filter.shape[2], stride)

        # perform convolution/cross correlation across entire input space
        conv_output_3d = list()
        for h in range(0, input.shape[1], stride):
            for w in range(0, input.shape[2], stride):
                region = input[:, h:h + filter_dim, w:w + filter_dim]
                if region.shape[1:] == (filter_dim, filter_dim):
                    region = region * filter
                    conv_output = np.sum(region, axis=(1, 2, 3))
                    conv_output_3d.append(conv_output)
                else:
                    break

        # reshape convolution output and add biases
        conv_output_3d = np.transpose(conv_output_3d).reshape(conv_output_shape)
        conv_output_3d += bias[:, None, None]
        return conv_output_3d

    @staticmethod
    def dact_fc_layer_outer(act_l, target):
        """
        Function to retrieve derivative of activations for final layer (e.g. softmax layer)
        :param act_l: (1d array) activations in layer
        :param target: (1d array) target output
        :return: (1d array) derivative of cost wrt activations for final layer
        """
        return act_l - target

    @staticmethod
    def dact_fc_layer_inner(w_l, dz_l, dact_l):
        """
        Function to retrieve derivative of activations for inner fully connected layers
        :param w_l: (2d array) weights in layer
        :param dz_l: (1d array) derivative of weighted sums in layer
        :param dact_l: (1d array) derivative of activations in layer
        :return: (1d array) derivative of cost wrt activations in new layer
        """
        dadact = w_l * dz_l
        return dadact @ dact_l

    @staticmethod
    def dcdw_fc(dz_l, act_l, dact_l):
        """
        Function to retrieve cost derivatives for fully connected layers
        :param dz_l: (1d array) derivative of weighted sums in layer
        :param act_l: (array-like) activations in layer
        :param dact_l: (1d array) derivative of cost wrt activations in layer
        :return: (tuple) cost derivatives for layer
        """
        # flatten if activations in layer is 3 dimensional tensor
        if len(act_l.shape) == 3:
            act_l = np.ravel(act_l)
        # calculate dcdw and dcdb for current layer
        dcdw_l = np.transpose([act_l]) @ [dz_l]
        dcdw_l = dcdw_l * np.transpose(dact_l)
        dcdb_l = dz_l * np.transpose(dact_l)
        return dcdw_l, dcdb_l, dact_l

    @staticmethod
    def apply_maxpool_mask(input_act, pool_filter_dim, pool_stride, maxpool_masks):
        """
        Function to apply pooling mask to input
        :param input_act: (3d tensor) input feature map
        :param pool_filter_dim: (int) filter dimensions
        :param pool_stride: (int) stride
        :param maxpool_masks: (3d tensor) pooling mask for layer
        :return: (3d tensor) updated pooling mask
        """
        for h in range(0, maxpool_masks.shape[1], pool_stride):
            for w in range(0, maxpool_masks.shape[2], pool_stride):
                region = maxpool_masks[:, h:h + pool_filter_dim, w:w + pool_filter_dim]
                if region.shape[1:] == (pool_filter_dim, pool_filter_dim):
                    loc = np.array(np.where(region == 1))
                    region[loc[0], loc[1, :], loc[2, :]] = input_act[:, h // pool_stride, w // pool_stride]
                else:
                    break
        return maxpool_masks

    @staticmethod
    def apply_padding(input, padding, filter_dim, stride):
        """
        Function to apply padding to input
        :param input: (3d tensor) input feature map
        :param padding: (string) type of padding
        :param filter_dim: (int) filter dimensions
        :param stride: (int) stride
        :return: (3d tensor) input with specified padding applied
        """
        if padding == 'valid':
            return input
        pad = (filter_dim - stride) // 2
        pad_tup = ((0, 0), (pad, pad), (pad, pad))
        return np.pad(input, pad_width=pad_tup, mode='constant')

    @staticmethod
    def dact_conv_l(input_act, padding, stride, filter, dz, dact):
        """
        Function to retrieve derivative of activations for convolutional layer
        :param input_act: (3d tensor) input feature map (activations in layer)
        :param padding: (string) type of padding
        :param stride: (int) stride
        :param filter: (4d tensor) filters in layer
        :param dz: (3d tensor) derivative of weighted sums in layer
        :param dact: (3d tensor) derivative of cost wrt activations in layer
        :return: (3d tensor) derivative of cost wrt activations in new layer
        """
        # apply padding if necessary
        filter_dim = filter.shape[2]
        input_act = Layers.apply_padding(input_act, padding, filter_dim, stride)
        # multiply dcdact by dz (derivative of weighted sums)
        dact = dact * dz
        # create a tensor with same shape as input
        input_act = np.full(input_act.shape, 1)

        # calculate dcdact for current layer
        dact_l = np.zeros((filter.shape[0], filter.shape[1], input_act.shape[1], input_act.shape[2]))
        for h in range(0, dact_l.shape[2], stride):
            for w in range(0, dact_l.shape[3], stride):
                region = input_act[:, h:h + filter_dim, w:w + filter_dim]
                if region.shape[1:] == (filter_dim, filter_dim):
                    region = filter * dact[:, h // stride, w // stride][:, None, None, None]
                    dact_l[:, :, h:h + filter_dim, w:w + filter_dim] += region
                else:
                    break

        # sum up all dcdact sheets across entire layer
        dact_l = np.sum(dact_l, axis=0)
        # if padding is same, remove padding from dcdact
        if padding == 'same':
            pad = (filter_dim - stride) // 2
            dact_l = dact_l[:, pad:-pad, pad:-pad]
        return dact_l

    @staticmethod
    def dcdw_conv_l(input_act, padding, stride, filter_dim, dz, dact_l):
        """
        Function to retrieve cost derivatives for convolutional layer
        :param input_act: (3d tensor) input feature map (activations in layer)
        :param padding: (string) type of padding
        :param stride: (int) stride
        :param filter_dim: (int) filter dimensions
        :param dz: (3d tensor) derivative of weighted sums in layer
        :param dact_l: (3d tensor) derivative of cost wrt activations in layer
        :return: (tuple) cost derivatives for layer
        """
        # apply padding if necessary
        input_act = Layers.apply_padding(input_act, padding, filter_dim, stride)
        # multiply dcdact of current layer with dz
        dz = dz * dact_l
        # calculate dcdb for current layer
        dcdb_l = np.sum(dz, axis=(1, 2))

        # calculate dcdw for current layer
        dcdw_l = 0
        for h in range(0, input_act.shape[1], stride):
            for w in range(0, input_act.shape[2], stride):
                region = input_act[:, h:h + filter_dim, w:w + filter_dim]
                if region.shape[1:] == (filter_dim, filter_dim):
                    dcdw_l += np.tensordot(dz[:, h // stride, w // stride], region, axes=0)
                else:
                    break
        return dcdw_l, dcdb_l, dact_l
