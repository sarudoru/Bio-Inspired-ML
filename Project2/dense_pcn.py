'''dense_pcn.py
Densely connected predictive coding network (PCN)
Jacob Petty, Saad Khan, and Sardor Nodirov
CS 443: Bio-Inspired Learning
'''
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from IPython import display
import math

import network
from dense_pcn_layer import InputPCNLayer, DensePCNLayer, OutputPCNLayer

class DensePCN(network.DeepNetwork):
    '''Predictive coding network with any number of densely connected layers. Uses linear activations throughout.
    '''
    def __init__(self, input_feats_shape, C, hidden_units=(256,), wt_scale=1e-2, gamma_lr=0.1,
                 train_num_steps=20, test_num_steps=10):
        '''DensePCN constructor. Builds the PCN, from input to output layers, and connects successive layers
        bidirectionally to one another.

        Parameters:
        -----------
        input_feats_shape: tuple of ints.
            The shape of input data WITHOUT the batch dimension.
            Example: If the input are 32x32 RGB images, input_feats_shape=(32, 32, 3).
        C: int.
            Number of classes in the dataset.
        hidden_units: tuple of ints.
            The number of units in each hidden layer of the PCN.
            Example: (256, 128) means the PCN has 2 hidden layers, the 1st with 256 hidden units and the 2nd hidden
            layer has 128 units.
        wt_scale: float.
            The standard deviation of the layer weights when initialized according to a standard normal distribution
            ('normal' method).
        gamma_lr: float.
            The strength with which the predictive feedforward and feedback signals affect the layer's evolving state.
        train_num_steps: int.
            Number of steps/iterations/sweeps to use when updating the state of each layer in the net during training.
        test_num_steps: int.
            Number of steps/iterations/sweeps to use when updating the state of each layer in the net during inference.

        TODO:
        1. Call the superclass constructor to handle any overlapping fields.
        2. Build out the PCN — input layer, the prescribed number of hidden layers, and the output layer.
        Connect up the layers appropriately. This means the input layer has a next connection (but not a prev one),
        the hidden layers are connected to the prev and next layers, and the ouput layer only has a prev connection.
        3. Keep track of your layers by inserting them into self.layers and assign the output layer to self.output_layer
        to make sure your network works with your existing `DeepNetwork` and `Layer` code.
        4. Create instance variables for parameters as needed.
        '''
        self.layers = []
        super().__init__(input_feats_shape)
        self.C = C
        self.hidden_units = hidden_units
        self.wt_scale = wt_scale
        self.gamma_lr = gamma_lr
        self.train_num_steps = train_num_steps
        self.test_num_steps = test_num_steps

        input_layer = InputPCNLayer(name="InputLayer", units=math.prod(self.input_feats_shape), gamma_lr=self.gamma_lr)

        num_hidden_units = len(self.hidden_units)
        for i in range(num_hidden_units):
            if i == 0:
                first_hidden = DensePCNLayer(name=f"PredLayer_{i}", units=self.hidden_units[i], wt_scale=self.wt_scale,
                                           prev_layer_or_block=input_layer, gamma_lr=self.gamma_lr)
                input_layer.set_next_layer(first_hidden)
                self.layers.append(input_layer)
                self.layers.append(first_hidden)
            else:
                new_hidden = DensePCNLayer(name=f"PredLayer_{i}", units=self.hidden_units[i], wt_scale=self.wt_scale,
                                           prev_layer_or_block=self.layers[i], gamma_lr=self.gamma_lr)
                self.layers[i].set_next_layer(new_hidden)
                self.layers.append(new_hidden)
        
        self.output_layer = OutputPCNLayer(name="OutputLayer", units=self.C, wt_scale=self.wt_scale, prev_layer_or_block=self.layers[-1], gamma_lr=self.gamma_lr)
        self.layers[-1].set_next_layer(self.output_layer)
        self.layers.append(self.output_layer)

    def set_test_num_steps(self, num_steps):
        '''Set method to update the number of steps/iterations/sweeps used by the PCN during inference.

        Parameters:
        -----------
        num_steps: int.
            Number of steps/iterations/sweeps to use by the PCN during inference.
        '''
        self.test_num_steps = num_steps

    def __call__(self, x):
        '''Perform the forward pass through the network.

        Parameters:
        -----------
        x: tf.float32 tensor. shape=(B, M).
            Mini-batch of data.

        Returns:
        --------
        tf.float32 tensor. shape=(B, C).
            netAct produced by the output layer.
        '''
        for layer in range(len(self.layers)):
            if layer == 0:
                current_net_act = self.layers[layer].__call__(x)
            else:
                current_net_act = self.layers[layer].__call__(current_net_act)

        return current_net_act

    def update_states(self, num_steps, x_batch, y_batch=None):
        '''Performs `num_steps` forward sweeps through the network to update the state in each successive layer.

        Parameters:
        -----------
        num_steps: int.
            Number of forward sweeps to perform in which each layer's state is updated.
        x_batch: tf.float32 tensor. shape=(B, M).
            Mini-batch of data.
        y_batch: tf.int32 tensor or None. shape=(B,).
            Int-coded labels of each sample in the mini-batch.

        TODO:
        1. Reset the state in all layers (to start fresh when processing each new mini-batch).
        2. Perform a forward pass through the network (to initialize all the layer states).
        3. Configure the output layer's clamping based on whether we are currently training or not. When we train, we
        clamp (i.e. fix) the output neurons that code the correct classes to have 1 states and set the remainder of the
        neuron states to 0s. If we are not training, we keep the output layer unclamped so that the output neuron states
        can evolve with those produced by the rest of the network.
        4. Iteratively update the state in each network layer from input → output layer.
        '''
        B, M = x_batch.shape
        
        # Step 1: Reset all states once at the beginning
        for layer in self.layers:
            layer.reset_state(B)
        
        # Step 2: Perform initial forward pass to initialize all layer states
        self.__call__(x_batch)
        
        # Step 3: Configure output layer clamping based on whether we are in training mode
        if y_batch is not None:  # Training mode
            # Clamp output layer and set its state to one-hot encoded labels
            self.output_layer.clamp_state()
            y_one_hot = tf.one_hot(y_batch, depth=self.C)
            self.output_layer.set_state(tf.cast(y_one_hot, tf.float32))
        else:  # Inference mode
            # Unclamp output layer so state can evolve
            self.output_layer.unclamp_state()
        
        # Keep input layer clamped so the mini-batch data is not modified
        self.layers[0].clamp_state()
        
        # Step 4: Iteratively update states in each layer
        for step in range(num_steps):
            for layer in self.layers:
                layer.update_state()

    def loss(self):
        '''Computes the loss for the current minibatch based on the states store in each network layer.

        Returns:
        -----------
        float.
            The loss.

        TODO:
        1. Compute the loss that the user specified when calling compile. We are assuming this will be 'predictive'
        (for predictive loss) for the PCN. Refer to the equation in the notebook for a refresher on predictive loss.
        2. Throw an error if the the user specified loss is not supported.
        '''
        if self.loss_name == 'predictive':
            total_loss = 0.0
            # Loop through all layers except input layer (start from index 1)
            for layer in self.layers[1:]:
                # Get prediction error for this layer
                pred_error = layer.prediction_error()
                # Square the errors
                squared_errors = tf.square(pred_error)
                # Sum across neurons for each sample
                sum_per_sample = tf.reduce_sum(squared_errors, axis=-1)
                # Average across batch and multiply by 0.5
                layer_loss = 0.5 * tf.reduce_mean(sum_per_sample)
                # Accumulate across layers
                total_loss += layer_loss
            return total_loss
        else:
            raise ValueError(f'Unknown loss function {self.loss_name}')

    def train_step(self, x_batch, y_batch):
        '''Completely process a single mini-batch of data during training. This should mirror your version from
        `DeepNetwork`, except for the following changes:
        1. Call your method to iteratively update the network layer states instead of performing a forward pass.
        2. Adapt how you compute the loss.

        Returns:
        --------
        float.
            The loss.
        '''
        self.set_layer_training_mode(is_training=True)
        with tf.GradientTape() as tape:
            self.update_states(self.train_num_steps, x_batch, y_batch)
            loss = self.loss()
        self.update_params(tape, loss)
        return loss

    def test_step(self, x_batch, y_batch, num_steps):
        '''Completely process a single mini-batch of data during test/validation time. This should mirror your version
        from `DeepNetwork`, except for the following changes:
        1. Call your method to iteratively update the network layer states instead of performing a forward pass.
        2. Adapt how you compute the loss.
        3. Obtain the predicted classes based on the output layer final STATES computed from the current mini-batch.
        The predicted class is the one coded by the output neuron with the largest state value.
        4. Compute the accuracy of the predictions.

        Parameters:
        -----------
        x_batch: tf.constant. tf.float32s. shape=(B, M).
            A single mini-batch of data packaged up by the fit method.
        y_batch: tf.constant. tf.ints32. shape=(B,).
            int-coded labels of samples in the mini-batch.

        Returns:
        --------
        float.
            The accuracy.
        float.
            The loss.

        NOTE:
        - You might run into annoying dtype casting issues depending on how you implemented your accuracy method.
        I suggest making sure the predicted classes are cast to tf.int32 before calling accuracy. This should make
        the dtype of the predicted classes the same as in the previous project.
        '''
        self.set_layer_training_mode(is_training=False)
        self.update_states(num_steps, x_batch)
        loss = self.loss()
        # Get predictions from output layer state (not net activation)
        output_state = self.output_layer.get_state()
        y_pred = tf.argmax(output_state, axis=-1, output_type=tf.int32)
        acc = self.accuracy(y_batch, y_pred)
        return acc, loss

    def evaluate(self, x, y, batch_sz=64):
        '''Evaluates the accuracy and loss on the data `x` and labels `y`. Breaks the dataset into mini-batches for you
        for efficiency.

        This method is provided to you, so you should not need to modify it.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, M).
            The complete dataset or one of its splits (train/val/test/dev).
        y: tf.constant. tf.ints32. shape=(B,).
            int-coded labels of samples in the complete dataset or one of its splits (train/val/test/dev).
        batch_sz: int.
            The batch size used to process the provided dataset. Larger numbers will generally execute faster, but
            all samples (and activations they create in the net) in the batch need to be maintained in memory at a time,
            which can result in crashes/strange behavior due to running out of memory.
            The default batch size should work fine throughout the semester and its unlikely you will need to change it.

        Returns:
        --------
        float.
            The accuracy.
        float.
            The loss.
        '''
        # Set the mode in all layers to the non-training mode
        self.set_layer_training_mode(is_training=False)

        # Make sure the mini-batch size isn't larger than the number of available samples
        N = len(x)
        if batch_sz > N:
            batch_sz = N

        num_batches = N // batch_sz

        # Make sure the mini-batch size is positive...
        if num_batches < 1:
            num_batches = 1

        # Process the dataset in mini-batches by the network, evaluating and avging the acc and loss across batches.
        loss = acc = 0
        for b in range(num_batches):
            curr_x = x[b*batch_sz:(b+1)*batch_sz]
            curr_y = y[b*batch_sz:(b+1)*batch_sz]

            curr_acc, curr_loss = self.test_step(curr_x, curr_y, self.test_num_steps)
            acc += curr_acc
            loss += curr_loss
        acc /= num_batches
        loss /= num_batches

        return acc, loss

    def dream_input(self, class_names, num_steps=150, image_dims=(28, 28, 1), input_stddev=1e-2, n_plot_rows=2,
                    eps=1e-8):
        '''Leverages the generative capabilities of the PCN to 'dream' up images that are the expected inputs for each
        output layer class neuron.

        NOTE: Much of this method is provided. Fill in parts of the method based on the inline TODO instructions.

        Parameters:
        -----------
        class_names: Python list of str.
            Names of each class in the dataset.
        num_steps: int.
            Number of test steps to use when dreaming.
        image_dims: tuple of ints. format: (Iy, Ix, n_chans).
            Expanded shape of single images in the dataset.
        input_stddev: float.
            Standard deviation to use when generating random noise in the placeholder for the dreamed input images.
        n_plot_rows: int.
            Number of rows of images in the plot that shows the generated dream image for each class / output neuron.
        eps: float.
            Fudge factor to prevent division by 0 when normalizing the generated images for plotting.
        '''
        M = self.input_feats_shape[0]
        C = self.output_layer.get_num_units()
        n_plot_cols = C // n_plot_rows

        fig, axes = plt.subplots(nrows=n_plot_rows, ncols=n_plot_cols, figsize=(12, 5))
        axes = axes.ravel()

        # Create all possible digits as separate samples in curr mini-batch
        generated_inputs = tf.random.normal(shape=(C, M), stddev=input_stddev)

        # We want to generate one dream image per output neuron. To do this, we create the states for the output layer
        # where the dreaming neuron j has state 1 and the rest have 0 for dream image j.
        # For example, this looks like:
        # [1, 0, 0]
        # [0, 1, 0]
        # [0, 0, 1]
        # if there are 3 classes and so 3 dream images to generate.
        one_hot_vecs = tf.eye(C)

        # TODO 1: Before we dream, reset all the states in the net
        for layer in self.layers:
            layer.reset_state(C)

        # TODO 2: Configure the input layer for dreaming: set the initial state to the random noise patterns and unclamp
        # the layer so the state can evolve.
        self.layers[0].set_state(generated_inputs)
        self.layers[0].unclamp_state()

        # TODO 3: Configure the output layer for dreaming: set the state of each the dreaming neuron to 1 and the rest
        # to 0 and clamp the layer so the network dynamics do not change the output layer state.
        self.output_layer.set_state(one_hot_vecs)
        self.output_layer.clamp_state()


        # Prepare plots for fast updating
        img_objs = []
        for i in range(C):
            img_objs.append(axes[i].imshow(np.zeros(image_dims), cmap='gray', vmin=0, vmax=1))
            axes[i].set_title(f'Class {class_names[i]}')
            axes[i].axis('off')

        # We want to do N_steps prediction passes for each class one hot vector
        # This is somewhat like update_states but we DO NOT want to do the forward pass
        # because the input is random noise and we don't care about the net_acts to it.
        # We want this to be 100% top-down driven.
        # Now evolve the states of each layer driven by the output neuron
        for t in range(num_steps):
            # Update the state in each layer for some number of steps to minimize the prediction error
            # We do this in reverse order for speed
            for layer in reversed(self.layers):
                layer.update_state()

            # Animate
            for i in range(C):
                # Reshape the state for the specific digit in the batch
                digit_pixels = tf.reshape(self.layers[0].get_state()[i], image_dims)
                # Min-max normalize for visualization
                min = tf.reduce_min(digit_pixels)
                max = tf.reduce_max(digit_pixels)
                digit_pixels = (digit_pixels - min) / (max - min + eps)
                img_objs[i].set_data(digit_pixels.numpy())

            # Set the suptitle as the frame number
            fig.suptitle(f'Frame {t}')

            display.clear_output(wait=True)
            display.display(fig)
            time.sleep(0.1)

        plt.close()

    def complete_input(self, x_batch, x_mask, y_batch, num_steps=150, image_dims=(28, 28, 1), n_plot_rows=2, eps=1e-8):
        '''Leverages the generative capabilities of the PCN to 'complete' masked portions of images based on the
        top-down prediction signals developed by the network.

        NOTE: Much of this method is provided. Fill in parts of the method based on the inline TODO instructions.

        Parameters:
        -----------
        x_batch: tf.constant. tf.float32s. shape=(B, M).
            A single mini-batch of masked images.
        x_mask: tf.constant. tf.float32s. shape=(B, M).
            The occlusion masks that specify where pixels in each mini-batch image are blanked out.
            Mask format: 0 if corresponding pixel in `x_batch` images is blanked out. 1 if corresponding pixel in
            `x_batch` images contains image information.
        y_batch: tf.constant. tf.ints32. shape=(B,).
            int-coded labels of samples in the mini-batch.
        image_dims: tuple of ints. format: (Iy, Ix, n_chans).
            Expanded shape of single images in the dataset.
        n_plot_rows: int.
            Number of rows of images in the plot that shows each completed image in the mini-batch.
        eps: float.
            Fudge factor to prevent division by 0 when normalizing the completed images for plotting.
        '''
        B = len(x_batch)
        C = self.output_layer.get_num_units()
        n_plot_cols = B // n_plot_rows

        fig, axes = plt.subplots(nrows=n_plot_rows, ncols=n_plot_cols, figsize=(12, 5))
        axes = axes.ravel()

        # Create one hot coding of every class in the mini-batch
        yh_batch = tf.one_hot(y_batch, C)

        # TODO 1: Before we fill in image detail, reset all the states in the net
        for layer in self.layers:
            layer.reset_state(B)

        # TODO 2: Configure the input layer for completion: set the initial state to the masked mini-batch,
        # make the mask available to the layer, and unclamp the layer.
        self.layers[0].set_state(x_batch)
        self.layers[0].set_mask(x_mask)
        self.layers[0].unclamp_state()

        # TODO 3: Configure the output layer for completion: set the state to the one-hot class labels and clamp the
        # state
        self.output_layer.set_state(yh_batch)
        self.output_layer.clamp_state()


        # Prepare plots for fast updating
        img_objs = []
        for i in range(C):
            img_objs.append(axes[i].imshow(np.zeros(image_dims), cmap='gray', vmin=0, vmax=1))
            axes[i].set_title(f'Class {y_batch[i]}')
            axes[i].axis('off')

        # We want to do N_steps prediction passes for each class one hot vector
        # This is somewhat like update_states but we DO NOT want to do the forward pass
        # because the input is random noise and we don't care about the net_acts to it.
        # We want this to be 100% top-down driven.
        # Now evolve the states of each layer driven by the output neuron
        for t in range(num_steps):
            # Update the state in each layer for some number of steps to minimize the prediction error
            # We do this in reverse order for speed
            for layer in reversed(self.layers):
                layer.update_state()

            # Animate
            for i in range(C):
                # Reshape the state for the specific digit in the batch
                digit_pixels = tf.reshape(self.layers[0].get_state()[i], image_dims)
                # Min-max normalize for visualization
                min = tf.reduce_min(digit_pixels)
                max = tf.reduce_max(digit_pixels)
                digit_pixels = (digit_pixels - min) / (max - min + eps)
                img_objs[i].set_data(digit_pixels.numpy())

            # Set the suptitle as the frame number
            fig.suptitle(f'Frame {t}')

            display.clear_output(wait=True)
            display.display(fig)
            time.sleep(0.1)

        plt.close()

