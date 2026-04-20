'''conv_layers.py
Convolutional layers in a predictive coding network (PCN)
Jacob Petty, Saad Khan, and Sardor Nodirov
CS 443: Bio-Inspired Learning
'''
import tensorflow as tf

import layers


class Conv2D(layers.Layer):
    '''A 2D convolutional layer'''
    def __init__(self, name, units, kernel_size=(1, 1), strides=1, activation='relu', wt_scale=1e-3,
                 prev_layer_or_block=None, wt_init='normal', do_group_norm=False):
        '''Conv2D layer constructor.

        Parameters:
        -----------
        name: str.
            Human-readable name for the current layer (e.g. Conv2D_0). Used for debugging and printing summary of net.
        units: ints.
            Number of convolutional filters/units (K).
        kernel_size: int or tuple. len(kernel_size)=2.
            The horizontal and vertical extent (pixels) of the convolutional filters.
            These will always be the same. For example: (2, 2), (3, 3), etc.
            If user passes in an int, we convert to a tuple. Example: 3 → (3, 3)
        strides: int.
            The horizontal AND vertical stride of the convolution operation. These will always be the same.
            By convention, we use a single int to specify both of them.
        activation: str.
            Name of the activation function to apply in the layer.
        wt_scale: float.
            The standard deviation of the layer weights/bias when initialized according to a standard normal
            distribution ('normal' method).
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.
        wt_init: str.
            The method used to initialize the weights/biases. Options: 'normal', 'he'.
            NOTE: Ignore until instructed otherwise.
        do_group_norm. bool:
            Whether to do group normalization in this layer.
            NOTE: Ignore until instructed otherwise.

        TODO: Set the parameters as instance variables. Call the superclass constructor to handle setting instance vars
        the child has in common with the parent class.
        '''

        # Keep me. You can specify kernel size as an int, but this snippet converts to tuple be explicitly account for
        # kernel width and height.
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        super().__init__(name, activation, prev_layer_or_block, do_group_norm)
        self.units = units
        self.kernel_size = kernel_size
        self.strides = strides
        self.wt_scale = wt_scale
        self.wt_init = wt_init

    def has_wts(self):
        '''Returns whether the Conv2D layer has weights. This is always true so always return... :)'''
        return True

    def init_params(self, input_shape):
        '''Initializes the Conv2D layer's weights and biases.

        Parameters:
        -----------
        input_shape: Python list. len(input_shape)=4.
            The anticipated shape of mini-batches of input that the layer will process: (B, Iy, Ix, K1).
            K1 is the number of units/filters in the previous layer.

        NOTE: Remember to set your wts/biases as tf.Variables so that we can update the values in the network graph
        during training.
        '''
        N, I_y, I_x, n_chans = input_shape
        if self.wt_init == 'he':
            # He/Kaiming initialization: std = gain / sqrt(fan_in)
            gain = self.get_kaiming_gain()
            fan_in = self.kernel_size[0] * self.kernel_size[1] * n_chans
            wt_std = gain / tf.sqrt(tf.cast(fan_in, tf.float32))
        else:
            # Normal initialization with user-specified scale
            wt_std = self.wt_scale

        # Weight shape: (kernel_height, kernel_width, n_chans, units)
        self.wts = tf.Variable(
            tf.random.normal(shape=(self.kernel_size[0], self.kernel_size[1], n_chans, self.units), stddev=wt_std),
            trainable=True, name="wts"
        )
        self.b = tf.Variable(tf.zeros(shape=(self.units,)), trainable=True, name="bias")


    def compute_net_input(self, x):
        '''Computes the net input for the current Conv2D layer. Uses SAME boundary conditions.

        Parameters:
        -----------
        x: tf.float32 tensor. shape=(B, Iy, Ix, K1).
            Input from the layer beneath in the network. K1 is the number of units in the previous layer.

        Returns:
        --------
        tf.float32 tensor. shape=(B, Iy, Ix, K2).
            The net_in. K2 is the number of units in the current layer.

        TODO:
        1. This layer uses lazy initialization. This means that if the wts are currently None when we enter this method,
        we should call `init_params` to initialize the parameters!
        2. Compute the convolution using TensorFlow's conv2d function. You can leave the dilations and data_format
        keyword arguments to their default values / you do not need to specify these parameters.

        Helpful link: https://www.tensorflow.org/api_docs/python/tf/nn/conv2d

        NOTE: Don't forget the bias!
        '''
        # Lazy initialization: if weights haven't been created yet, create them now
        if self.wts is None:
            self.init_params(x.shape)

        # tf.nn.conv2d expects strides as a 4-element list: [batch, height, width, channels]
        conv_strides = [1, self.strides, self.strides, 1]
        net_in = tf.nn.conv2d(x, self.wts, strides=conv_strides, padding='SAME') + self.b

        # # Apply group normalization if enabled
        # if self.do_group_norm:
        #     net_in = self.compute_group_norm(net_in)

        return net_in

    def compute_group_norm(self, net_in, eps=0.001):
        '''Computes group normalization for the input tensor. Group normalization normalizes the activations among
        groups of neurons in a layer for each data point independently.

        Uses `self.num_groups` groups of neurons to perform the normalization. If the user has not set a value for
        `self.num_groups` for the current layer, we set it to the number of units in the layer divided by 8 (rounded).
        In this case when the user did not specify `self.num_groups`, we do not allow the number of groups to exceed
        8.

        (Ignore until later in the semester)

        Parameters:
        -----------
        x: tf.float32 tensor. shape=(B, Iy, Ix, K).
            Input tensor to be normalized.
        eps: float.
            A small constant added to the standard deviation to prevent division by zero. Default is 0.001.

        Returns:
        --------
        tf.float32 tensor. shape=(B, Iy, Ix, K).
            The normalized tensor with the same shape as the input tensor.

        NOTE: This method should handle the scaling and shifting.
        '''
        B, Iy, Ix, K = net_in.shape

        # Set default number of groups if not specified by the user
        if self.num_groups is None:
            self.num_groups = min(round(K / 8), 8)

        # Compute the number of channels per group
        group_size = K // self.num_groups

        # Reshape from (B, Iy, Ix, K) to (B, Iy, Ix, num_groups, group_size)
        x = tf.reshape(net_in, [B, Iy, Ix, self.num_groups, group_size])

        # Compute mean and variance along the last axis (within each group)
        mean = tf.reduce_mean(x, axis=[1, 2, 4], keepdims=True)
        var = tf.reduce_mean(tf.math.squared_difference(x, mean), axis=[1, 2, 4], keepdims=True)

        # Normalize: (x - mean) / sqrt(var + eps)
        x_norm = (x - mean) / (tf.sqrt(var) + eps)

        # Reshape back to original shape (B, Iy, Ix, K)
        x_norm = tf.reshape(x_norm, [B, Iy, Ix, K])

        # Initialize gain and bias (scaling and shifting) if they haven't been set
        if self.gn_gain is None:
            self.gn_gain = tf.Variable(tf.ones([K]), trainable=True, name="gn_gain")
        if self.gn_bias is None:
            self.gn_bias = tf.Variable(tf.zeros([K]), trainable=True, name="gn_bias")

        # Apply scaling and shifting
        return x_norm * self.gn_gain + self.gn_bias

    def __str__(self):
        '''This layer's "ToString" method. Feel free to customize if you want to make the layer description fancy,
        but this method is provided to you. You should not need to modify it.
        '''
        return f'Conv2D layer output({self.layer_name}) shape: {self.output_shape}'


class MaxPool2D(layers.Layer):
    '''A 2D maxpooling layer.'''
    def __init__(self, name, pool_size=(2, 2), strides=1, prev_layer_or_block=None):
        '''MaxPool2D layer constructor.

        Parameters:
        -----------
        name: str.
            Human-readable name for the current layer (e.g. Drop_0). Used for debugging and printing summary of net.
        pool_size. tuple. len(pool_size)=2.
            The horizontal and vertical size of the pooling window.
            These will always be the same. For example: (2, 2), (3, 3), etc.
        strides. int.
            The horizontal AND vertical stride of the max pooling operation. These will always be the same.
            By convention, we use a single int to specify both of them.
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.
            Example (standard MLP): Input → Dense_Hidden → Dense_Output.
                The Dense_Output Layer object has `prev_layer_or_block=Dense_Hidden`.

        TODO: Set the parameters as instance variables. Call the superclass constructor to handle setting instance vars
        the child has in common with the parent class.
        '''

        # Keep me. You can specify pool window size as an int, but this snippet converts to tuple be explicitly account
        # for window width and height.
        if isinstance(pool_size, int):
            pool_size = (pool_size, pool_size)
                # Call parent constructor (no activation, no group norm for pooling)
        super().__init__(name, activation='linear', prev_layer_or_block=prev_layer_or_block, do_group_norm=False)

        self.pool_size = pool_size
        self.strides = strides
        

    def compute_net_input(self, x):
        '''Computes the net input for the current MaxPool2D layer.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, Iy, Ix, K1).
            Input from the layer beneath in the network. Should be 4D (e.g. from a Conv2D or MaxPool2D layer).
            K1 refers to the number of units/filters in the PREVIOUS layer.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, Iy, Ix, K2).
            The net_in. K2 refers to the number of units/filters in the CURRENT layer.

        TODO: Compute the max pooling using TensorFlow's max_pool2d function. You can leave the data_format
        keyword arguments to its default value.

        Helpful link: https://www.tensorflow.org/api_docs/python/tf/nn/max_pool2d
        '''
        ksize = [1, self.pool_size[0], self.pool_size[1], 1]
        strides = [1, self.strides, self.strides, 1]

        net_in = tf.nn.max_pool2d(
            x,
            ksize=ksize,
            strides=strides,
            padding='VALID'
        )

        return net_in
    def __str__(self):
        '''This layer's "ToString" method. Feel free to customize if you want to make the layer description fancy,
        but this method is provided to you. You should not need to modify it.
        '''
        return f'MaxPool2D layer output({self.layer_name}) shape: {self.output_shape}'


class Conv2DTranspose(Conv2D):
    '''2D transposed convolution layer'''
    def __init__(self, name, kernel_size=(1, 1), strides=1, activation='linear', wt_scale=1e-3,
                 prev_layer_or_block=None, wt_init='normal', do_group_norm=False):
        '''Conv2DTranspose layer constructor.

        Parameters:
        -----------
        name: str.
            Human-readable name for the current layer (e.g. Conv2DTranspose_0). Used for debugging and printing.
        kernel_size: int or tuple. len(kernel_size)=2.
            The horizontal and vertical extent (pixels) of the convolutional filters.
            These will always be the same. For example: (2, 2), (3, 3), etc.
            If user passes in an int, we convert to a tuple. Example: 3 → (3, 3)
        strides. int.
            The horizontal AND vertical stride of the convolution operation. These will always be the same.
            By convention, we use a single int to specify both of them.
        activation: str.
            Name of the activation function to apply in the layer.
        wt_scale: float.
            The standard deviation of the layer weights/bias when initialized according to a standard normal
            distribution ('normal' method).
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.
        wt_init: str.
            The method used to initialize the weights/biases. Options: 'normal', 'he'.
            NOTE: Ignore until instructed otherwise.
        do_group_norm. bool:
            Whether to do group normalization in this layer.
            NOTE: Ignore until instructed otherwise.

        TODO: Call the superclass constructor to handle setting instance vars the child has in common with the parent
        class.

        NOTE: The units will be set during lazy initialization so set units to None for the time being.
        '''
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
            
        super().__init__(name, units=None, kernel_size=kernel_size, strides=strides, activation=activation,
                         wt_scale=wt_scale, prev_layer_or_block=prev_layer_or_block, wt_init=wt_init, do_group_norm=do_group_norm)

    def init_params(self, input_shape):
        '''Initializes the Conv2D layer's weights and biases.

        Parameters:
        -----------
        input_shape: Python list. len(input_shape)=4.
            The anticipated shape of mini-batches of input that the layer will process: (B, Iy, Ix, K1).
            K1 is the number of units/filters in the previous layer.

        NOTE: Remember to set your wts/biases as tf.Variables so that we can update the values in the network graph
        during training.
        '''

        N, I_y, I_x, n_chans = input_shape
        if self.wt_init == 'he':
            # He/Kaiming initialization: std = gain / sqrt(fan_in)
            # For transposed convolution, fan_in is kernel_size * kernel_size * n_chans (input channels to the operation)
            gain = self.get_kaiming_gain()
            fan_in = self.kernel_size[0] * self.kernel_size[1] * n_chans
            wt_std = gain / tf.sqrt(tf.cast(fan_in, tf.float32))
        else:
            # Normal initialization with user-specified scale
            wt_std = self.wt_scale

        # Weight shape for conv2d_transpose: (kernel_height, kernel_width, output_channels, input_channels)
        # output_channels is self.units (number of units in the layer below)
        # input_channels is n_chans (number of units in the layer above, i.e., input to this layer)
        self.wts = tf.Variable(
            tf.random.normal(shape=(self.kernel_size[0], self.kernel_size[1], self.units, n_chans), stddev=wt_std),
            trainable=True, name="wts"
        )
        self.b = tf.Variable(tf.zeros(shape=(self.units,)), trainable=True, name="bias")


    def compute_net_input(self, x, units_prev):
        '''Computes the net input for the current Conv2D layer. Uses SAME boundary conditions.

        Parameters:
        -----------
        x: tf.float32 tensor. shape=(B, Iy, Ix, K).
            Input from the *current* layer. K is the number of units in the *current* layer.
        units_prev: int.
            Number of units in the *previous* layer beneath the current one.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, Iy, Ix, units_prev).
            The net_in. units_prev is the number of units in the *previous* layer beneath the current one.

        TODO:
        0. Because we do not know the units in the layer below when creating the Conv2DTranspose layer, self.units
        should be None during the first forward pass. If this is the case, make sure to set the units to `units_prev`
        (remember this layer is for sending feedback).
        1. This layer uses lazy initialization. This means that if the wts are currently None when we enter this method,
        we should call `init_params` to initialize the parameters!
        2. Compute the convolution using TensorFlow's conv2d_transpose function.
        You can leave the dilations and data_format keyword arguments to their default values / you do not need to
        specify these parameters.
        3. When computing the netIn output shape, remember to account for the stride. When stride >1, this has an
        UPSCALING effect on the image. For example for a 32x32 image that is processed with conv2d_transpose and
        stride 2, the final image size will be 64x64.

        Helpful link: https://www.tensorflow.org/api_docs/python/tf/nn/conv2d_transpose

        NOTE: Don't forget the bias!
        '''
        if self.units is None: 
            self.units = units_prev

        if self.wts is None:
            self.init_params(x.shape)

        # Compute output shape
        batch_sz = tf.shape(x)[0]
        in_h = tf.shape(x)[1]
        in_w = tf.shape(x)[2]
        
        # For 'SAME' padding, output spatial size is input spatial size * stride
        out_shape = [batch_sz, in_h * self.strides, in_w * self.strides, self.units]
        
        conv_strides = [1, self.strides, self.strides, 1]

        net_in = tf.nn.conv2d_transpose(x, self.wts, output_shape=out_shape, strides=conv_strides, padding='SAME')
        net_in = net_in + self.b
        
        # if self.do_group_norm:
        #     net_in = self.compute_group_norm(net_in)
            
        return net_in


    def __call__(self, x, units_prev):
        '''Do a forward pass thru the layer with mini-batch `x`.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, Iy, Ix, K)
            The input mini-batch computed in the current layer.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, Iy, Ix, K_prev)
            The activation computed on the current mini-batch.

        TODO: This should be the same as the one in Layer, except note the difference in method signature for
        compute_net_input.
        '''
        net_in = self.compute_net_input(x, units_prev)
        
        if self.do_group_norm and self.gn_gain is not None:
            net_in = self.compute_group_norm(net_in)
            
        activ = self.compute_net_activation(net_in)
        self.output_shape = activ.shape
        return activ

    def __str__(self):
        '''This layer's "ToString" method. Feel free to customize if you want to make the layer description fancy,
        but this method is provided to you. You should not need to modify it.
        '''
        return f'Conv2DTranspose layer output({self.layer_name}) shape: {self.output_shape}'
