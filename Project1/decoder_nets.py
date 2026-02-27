'''hebb_net.py
Artificial neural networks that decoder signals encoding in the activations of the Hebbian network.
Jacob Petty, Sardor Nodirov, and Saad Khan
CS 443: Bio-Inspired Machine Learning
Project 1: Hebbian Learning
'''
import tensorflow as tf

import network
from layers import Dense

class LinearDecoder(network.DeepNetwork):
    '''Linear Decoder network with the following architecture:

    Dense output layer (softmax activation).

    Uses standard cross-entropy loss.
    '''
    def __init__(self, input_feats_shape, C):
        '''LinearDecoder contructor

        Parameters:
        -----------
        input_feats_shape: tuple.
            The FLATTEN shape of input data WITHOUT the batch dimension.
            Example: If the input are 32x32 RGB images, input_feats_shape=(32*32*3,).
        C: int.
            Number of classes in the dataset.

        TODO: Build the network and create instance variables as needed.

        NOTE:
        1. You should make use of the parent's constructor to initialize common variables.
        2. The output layer for ANY `DeepNetwork` here and going forward should be assigned to the variable
        self.output_layer.
        '''
        super().__init__(input_feats_shape)
        self.output_layer = Dense('Output Layer', C, activation='softmax', prev_layer_or_block=None)
        

    def __call__(self, x):
        '''Do a forward pass thru the network with mini-batch `x`.

        Parameters:
        -----------
        x: tf.float32 tensor. shape=(B, M)
            The input mini-batch computed in the current layer.

        Returns:
        --------
        tf.float32 tensor. shape=(B, M).
            The output layer activation computed on the current mini-batch.
        '''
        return self.output_layer(x)

class NonlinearDecoder(network.DeepNetwork):
    '''Nonlinear Decoder network proposed by Krotov & Hopfield with the following architecture:

    Dense output layer (tanh activation).

    Uses lp loss.
    '''
    def __init__(self, input_feats_shape, C, wt_scale=0.1, beta=0.0025, loss_exp=2.0):
        '''NonlinearDecoder constructor

        Parameters:
        -----------
        input_feats_shape: tuple.
            The FLATTEN shape of input data WITHOUT the batch dimension.
        C: int.
            Number of classes in the dataset.
        wt_scale: float.
            The standard deviation of the layer weights.
        beta: float.
            β hyperparameter inside the tanh activation function.
        loss_exp: float
            The exponent `m` in the lp loss function.
        '''
        super().__init__(input_feats_shape)
        self.output_layer = Dense('Output Layer', C, activation='tanh', 
                                   prev_layer_or_block=None, wt_scale=wt_scale)
        self.output_layer.set_tanh_beta(beta)
        self.output_layer.loss_exp = loss_exp

    def __call__(self, x):
        '''Do a forward pass thru the network with mini-batch `x`.

        Parameters:
        -----------
        x: tf.float32 tensor. shape=(B, M)
            The input mini-batch.

        Returns:
        --------
        tf.float32 tensor. shape=(B, C).
            The output layer activation computed on the current mini-batch.
        '''
        return self.output_layer(x)
