'''network.py
Deep neural network core functionality implemented with the low-level TensorFlow API.
Jacob Petty, Sardor Nodirov, and Saad Khan
CS 443: Bio-Inspired Learning
'''
import time
import numpy as np
import tensorflow as tf

from tf_util import arange_index

class DeepNetwork:
    '''The DeepNetwork class is the parent class for specific networks (e.g. LinearDecoder, NonlinearDecoder).
    '''
    def __init__(self, input_feats_shape):
        '''DeepNetwork constructor.

        Parameters:
        -----------
        input_feats_shape: tuple.
            The shape of input data WITHOUT the batch dimension.
            Example: If the input are 32x32 RGB images, input_feats_shape=(32, 32, 3).

        TODO: Set instance variables for the parameters passed into the constructor.
        '''
        # Keep these instance vars:
        self.loss_name = None
        self.output_layer = None
        self.all_net_params = []

        self.input_feats_shape = input_feats_shape

    def compile(self, loss='cross_entropy', lr=1e-3, print_summary=True):
        '''Compiles the neural network to prepare for training.

        This involves performing the following tasks:
        1. Storing instance vars for the loss function and optimizer that will be used when training.
        2. Initializing the optimizer.
        3. Doing a "pilot run" forward pass with a single fake data sample that has the same shape as those that will be
        used when training. This will trigger each weight layer's lazy initialization to initialize weights, biases, and
        any other parameters.
        4. (Optional) Print a summary of the network architecture (layers + shapes) now that we have initialized all the
        layer parameters and know what the shapes will be.
        5. Get references to all the trainable parameters (e.g. wts, biases) from all network layers. This list will be
        used during backpropogation to efficiently update all the network parameters.

        Parameters:
        -----------
        loss: str.
            Loss function to use during training.
        lr: float.
            Learning rate used by the optimizer during training.
        print_summary: bool.
            Whether to print a summary of the network architecture and shapes of activations in each layer.

        TODO: Fill in the section below that should create the supported optimizer.
        Use the TensorFlow Keras Adam optimizer. Assign the optimizer to the instance variable named `opt`.
        '''
        self.loss_name = loss

        # Initialize optimizer
        #TODO: Fill this section in
        self.opt = tf.keras.optimizers.Adam(learning_rate=lr)

        # Do 'fake' forward pass through net to create wts/bias
        x_fake = self.get_one_fake_input()
        self(x_fake)

        # Initialize group norm vars
        self.init_groupnorm_params()

        # Print network arch
        if print_summary:
            self.summary()

        # Get reference to all net params
        self.all_net_params = self.get_all_params()

    def get_one_fake_input(self):
        '''Generates a fake mini-batch of one sample to forward through the network when it is compiled to trigger
        lazy initialization to instantiate the weights and biases in each layer.

        This method is provided to you, so you should not need to modify it.
        '''
        return tf.zeros(shape=(1, *self.input_feats_shape))

    def summary(self):
        '''Traverses the network backward from output layer to print a summary of each layer's name and shape.

        This method is provided to you, so you should not need to modify it.
        '''
        print(75*'-')
        layer = self.output_layer
        while layer is not None:
            print(layer)
            layer = layer.get_prev_layer_or_block()
        print(75*'-')

    def set_layer_training_mode(self, is_training):
        '''Sets the training mode in each network layer.

        Parameters:
        -----------
        is_training: bool.
            True if the network is currently in training mode, False otherwise.

        TODO: Starting with the output layer, traverse the net backward, calling the appropriate method to
        set the training mode in each network layer. Model this process around the summary method.
        '''
        layer = self.output_layer
        while layer is not None:
            layer.set_mode(is_training)
            layer = layer.get_prev_layer_or_block()

    def init_groupnorm_params(self):
        '''Initializes group norm related parameters in all layers that are using batch normalization.

        (Ignore until instructed later in the semester)

        TODO: Starting with the output layer, traverse the net backward, calling the appropriate method to
        initialize the group norm parameters in each network layer. Model this process around the summary method.
        '''
        pass

    def get_all_params(self, wts_only=False):
        '''Traverses the network backward from the output layer to compile a list of all trainable network paramters.

        This method is provided to you, so you should not need to modify it.

        Parameters:
        -----------
        wts_only: bool.
            Do we only collect a list of only weights (i.e. no biases or other parameters).

        Returns:
        --------
        Python list.
            List of all trainable parameters across all network layers.
        '''
        all_net_params = []

        layer = self.output_layer
        while layer is not None:
            if wts_only:
                params = layer.get_wts()

                if params is None:
                    params = []
                if not isinstance(params, list):
                    params = [params]
            else:
                params = layer.get_params()

            all_net_params.extend(params)
            layer = layer.get_prev_layer_or_block()
        return all_net_params

    def accuracy(self, y_true, y_pred):
        '''Computes the accuracy of classified samples. Proportion correct.

        Parameters:
        -----------
        y_true: tf.int32 tensor. shape=(B,).
            int-coded true classes.
        y_pred: tf.int32 tensor. shape=(B,).
            int-coded predicted classes by the network.

        Returns:
        -----------
        float.
            The accuracy in range [0, 1]

        Hint: tf.where might be helpful.
        '''
        return len(tf.where(y_pred == y_true)) / len(y_true)

    def predict(self, x, output_layer_net_act=None):
        '''Predicts the class of each data sample in `x` using the passed in `output_layer_net_act`.
        If `output_layer_net_act` is not passed in, the method should compute the output layer activations in order to
        perform the prediction.

        Parameters:
        -----------
        x: tf.float32 tensor. shape=(B, ...).
            Data samples
        output_layer_net_act: tf.constant. shape=(B, C) or None.
            Network activation.

        Returns:
        -----------
        tf.int32 tensor. shape=(B,).
            int-coded predicted class for each sample in the mini-batch.
        '''
        if output_layer_net_act is None:
            output_layer_net_act = self(x)
        return tf.argmax(output_layer_net_act, axis=-1, output_type=tf.int32)

    def loss(self, out_net_act, y, eps=1e-16):
        '''Computes the loss for the current minibatch based on the output layer activations `out_net_act` and int-coded
        class labels `y`.

        Parameters:
        -----------
        output_layer_net_act: tf.float32 tensor. shape=(B, C) or None.
            Net activation in the output layer for the current mini-batch.
        y: tf.int32 tensor. shape=(B,).
            int-coded true classes for the current mini-batch.

        Returns:
        -----------
        float.
            The loss.

        TODO:
        1. Compute the loss that the user specified when calling compile. As of the first week of Project 1,
        the only option that should be supported/implemented is 'cross_entropy' for general cross-entropy loss
        (you will add support for additional options in the future).
        2. Throw an error if the the user specified loss is not supported.

        NOTE: I would like you to implement cross-entropy loss "from scratch" here — i.e. using the equation provided
        in the notebook, NOT using a TF high level function. For your convenience, I am providing the `arange_index`
        function in tf_util.py that offers functionality that is similar to arange indexing in NumPy (which you cannot
        do in TensorFlow). Use it!
        '''
        if self.loss_name == 'cross_entropy':
            y_pred = arange_index(out_net_act, y)
            return -tf.reduce_mean(tf.math.log(y_pred + eps))
        elif self.loss_name == 'lp':
            # Get number of classes C
            C = int(out_net_act.shape[1])
            # Create one-hot encoding with -1 for "off" values and 1 for "on" values
            y_one_hot = tf.one_hot(y, C, on_value=1.0, off_value=-1.0)
            y_one_hot = tf.cast(y_one_hot, tf.float32)
            
            # Get loss exponent m
            loss_exp = self.output_layer.loss_exp if hasattr(self.output_layer, 'loss_exp') else 2.0
            loss_exp = tf.cast(loss_exp, tf.float32)
            
            # Compute L^p loss: mean of |yh - net_act|^m
            diff = tf.abs(y_one_hot - out_net_act)
            powered = tf.pow(diff, loss_exp)
            loss = tf.reduce_mean(powered)
            
            return loss
        else:
            raise ValueError(f'Unsupported loss function: {self.loss_name}')

    def update_params(self, tape, loss):
        '''Do backpropogation: have the optimizer update the network parameters recorded on `tape` based on the
        gradients computed of `loss` with respect to each of the parameters. The variable `self.all_net_params`
        represents a 1D list of references to ALL trainable parameters in every layer of the network
        (see compile method).

        This method is provided to you, so you should not need to modify it.

        Parameters:
        -----------
        tape: tf.GradientTape.
            Gradient tape object on which all the gradients have been recorded for the most recent forward pass.
        loss: tf.Variable. float.
            The loss computed over the current mini-batch.
        '''
        grads = tape.gradient(loss, self.all_net_params)
        self.opt.apply_gradients(zip(grads, self.all_net_params))

    def train_step(self, x_batch, y_batch):
        '''Completely process a single mini-batch of data during training. This includes:
        1. Performing a forward pass of the data through the entire network.
        2. Computing the loss.
        3. Updating the network parameters using backprop (via update_params method).

        Parameters:
        -----------
        x_batch: tf.float32 tensor. shape=(B, ...).
            A single mini-batch of data packaged up by the fit method.
        y_batch: tf.ints32 tensor. shape=(B,).
            int-coded labels of samples in the mini-batch.

        Returns:
        --------
        float.
            The loss.

        NOTE: Don't forget to record gradients on a gradient tape!
        '''
        with tf.GradientTape() as tape:
            out_net_act = self(x_batch)
            loss = self.loss(out_net_act, y_batch)
        self.update_params(tape, loss)
        return loss

    def test_step(self, x_batch, y_batch):
        '''Completely process a single mini-batch of data during test/validation time. This includes:
        1. Performing a forward pass of the data through the entire network.
        2. Computing the loss.
        3. Obtaining the predicted classes for the mini-batch samples.
        4. Compute the accuracy of the predictions.

        Parameters:
        -----------
        x_batch: tf.float32 tensor. shape=(B, ...).
            A single mini-batch of data packaged up by the fit method.
        y_batch: tf.ints32 tensor. shape=(B,).
            int-coded labels of samples in the mini-batch.

        Returns:
        --------
        float.
            The accuracy.
        float.
            The loss.
        '''
        out_net_act = self(x_batch)
        loss = self.loss(out_net_act, y_batch)
        y_pred = self.predict(x_batch, out_net_act)
        acc = self.accuracy(y_batch, y_pred)
        return acc, loss

    def fit(self, x, y, x_val=None, y_val=None, batch_size=128, max_epochs=10000, val_every=1, print_every=10,
            verbose=True, patience=999, lr_patience=999, lr_decay_factor=0.5, lr_max_decays=12):
        '''Trains the neural network on the training samples `x` (and associated int-coded labels `y`).

        Parameters:
        -----------
        x: tf.float32 tensor. shape=(N_train, Iy, Ix, n_chans) or (N_train, M).
            The data samples.
        y: tf.int32 tensor. shape=(N_train,).
            int-coded class labels
        x_val: tf.float32 tensor. shape=(N_val, Iy, Ix, n_chans) or (N_val, M).
            Validation set samples.
        y_val: tf.float32 tensor. shape=(N_val,).
            int-coded validation set class labels.
        batch_size: int.
            Number of samples to include in each mini-batch.
        max_epochs: int.
            Network should train no more than this many epochs.
            Why it is not just called `epochs` will be revealed in Week 2.
        val_every: int.
            How often (in epoches) to compute validation set accuracy and loss.
        verbose: bool.
            If `False`, there should be no print outs during training. Messages indicating start and end of training are
            fine.
        patience: int.
            Number of most recent computations of the validation set loss to consider when deciding whether to stop
            training early (before `max_epochs` is reached).
            NOTE: Ignore until instructed otherwise.
        lr_patience: int.
            Number of most recent computations of the validation set loss to consider when deciding whether to decay the
            optimizer learning rate.
            NOTE: Ignore until instructed otherwise.
        lr_decay_factor: float.
            A value between 0.0. and 1.0 that represents the proportion of the current learning rate that the learning
            rate should be set to. For example, 0.7 would mean the learning rate is set to 70% its previous value.
            NOTE: Ignore until instructed otherwise.
        lr_max_decays: int.
            Number of times we allow the lr to decay during training.
            NOTE: Ignore until instructed otherwise.

        Returns:
        -----------
        train_loss_hist: Python list of floats. len=num_epochs.
            Training loss computed on each training mini-batch and averaged across all mini-batchs in one epoch.
        val_loss_hist: Python list of floats. len=num_epochs//val_freq.
            Loss computed on the validation set every time it is checked (`val_every`).
        val_acc_hist: Python list of floats. len=num_epochs//val_freq.
            Accuracy computed on the validation every time it is checked  (`val_every`).
        e: int.
            The number of training epochs used.

        TODO:
        0. To properly handle Dropout layers in your network, set the mode of all layers in the network to train mode
        before the training loop begins.
        1. Process the data in mini-batches of size `batch_size` for each training epoch. Use the strategy recommended
        in CS343 for sampling the dataset randomly WITH replacement.
            NOTE: I suggest using NumPy to create a RNG (before the training loop) with a fixed random seed (of 0) to
            generate mini-batch indices. That way you can ensure that differing results you get across training runs are
            not just due to your random choice of samples in mini-batches. This should probably be your ONLY use of
            NumPy in `DeepNetwork`.
        2. Call `train_step` to handle the forward and backward pass on the mini-batch.
        3. Average and record training loss values across all mini-batches in each epoch (i.e. one avg per epoch).
        4. If we are at the end of an appropriate epoch (determined by `val_every`):
            - Check and record the acc/loss on the validation set.
            - Print out: current epoch, training loss, val loss, val acc
        5. Regardless of `val_every`, print out the current epoch number (and the total number). Use the time module to
        also print out the time it took to complete the current epoch. Try to print the time and epoch number on the
        same line to reduce clutter.

        NOTE:
        - The provided `evaluate` method (below) should be useful for computing the validation acc+loss ;)
        - `evaluate` kicks all the network layers out of training mode (as is required bc it is doing prediction).
        Be sure to bring the network layers back into training mode after you are doing computing val acc+loss.
        - There should be zero print outs when verbose is set to False.
        '''
        # Define loss tracking containers
        train_loss_hist = []
        val_loss_hist = []
        val_acc_hist = []
        recent_val_losses = [] 
        rolling_val_losses = []
        num_decays = 0

        N = int(x.shape[0])
        num_batches = N // batch_size
        if num_batches < 1:
            num_batches = 1

        self.set_layer_training_mode(is_training=True)

        total_start = time.time()

        for e in range(max_epochs):
            epoch_start = time.time()

            epoch_loss = 0
            for b in range(num_batches):
                batch_indices = tf.random.uniform(shape=(batch_size,), minval=0, maxval=N, dtype=tf.int32)
                x_batch = tf.gather(x, batch_indices)
                y_batch = tf.gather(y, batch_indices)

                loss = self.train_step(x_batch, y_batch)
                epoch_loss += loss

            avg_epoch_loss = epoch_loss / num_batches
            train_loss_hist.append(float(avg_epoch_loss))

            epoch_time = time.time() - epoch_start

            # Validation check
            val_acc = None
            val_loss = None
            if x_val is not None and y_val is not None and (e + 1) % val_every == 0:
                val_acc, val_loss = self.evaluate(x_val, y_val)
                val_loss_hist.append(float(val_loss))
                val_acc_hist.append(float(val_acc))
                self.set_layer_training_mode(is_training=True)
                
                # Check early stopping
                recent_val_losses, should_stop = self.early_stopping(recent_val_losses, float(val_loss), patience)

                if should_stop:
                    if verbose:
                        print(f'Early stopping triggered at epoch {e}')
                    break

                rolling_val_losses, should_decay = self.early_stopping(rolling_val_losses, float(val_loss), lr_patience)
                if should_decay and num_decays < lr_max_decays:
                    self.lr_step_decay(lr_decay_factor)
                    num_decays += 1

            # Print output matching expected format
            if verbose:
                val_acc_str = f'{val_acc:.4f}' if val_acc is not None else 'N/A'
                val_loss_str = f'{val_loss:.3f}' if val_loss is not None else 'N/A'
                print(f'Epoch {e}/{max_epochs-1}, Training loss {float(avg_epoch_loss):.3f}, Val loss {val_loss_str}, Val acc {val_acc_str}')
                print(f'Epoch {e} took: {epoch_time:.1f} secs')

        total_time = time.time() - total_start
        print(f'Training finished after {e} epochs in {total_time:.2f} seconds')
        print(f'Learning rate was decayed {num_decays} times.')
        return train_loss_hist, val_loss_hist, val_acc_hist, e

    def evaluate(self, x, y, batch_sz=64):
        '''Evaluates the accuracy and loss on the data `x` and labels `y`. Breaks the dataset into mini-batches for you
        for efficiency.

        This method is provided to you, so you should not need to modify it.

        Parameters:
        -----------
        x: tf.float32 tensor. shape=(N, Iy, Ix, n_chans).
            The complete dataset or one of its splits (train/val/test/dev).
        y: tf.constant. tf.ints32. shape=(N,).
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

            curr_acc, curr_loss = self.test_step(curr_x, curr_y)
            acc += curr_acc
            loss += curr_loss
        acc /= num_batches
        loss /= num_batches

        return acc, loss

    def early_stopping(self, recent_val_losses, curr_val_loss, patience):
        '''Helper method used during training to determine whether training should stop before the maximum number of
        training epochs is reached based on the most recent loss values computed on the validation set
        (`recent_val_losses`) the validation loss on the current epoch (`curr_val_loss`) and `patience`.

        (Week 3)

        The logic:
        1. When training begins, the recent history of validation loss values `recent_val_losses` is empty (i.e. `[]`).
        We always insert the current val loss.
        2. We always enforce that the length of `recent_val_losses` should not exceed `patience`+1
            (the val losses on the most recent `patience` epochs + the val loss on the current epoch).
            The recent history of validation loss values (`recent_val_losses`) is assumed to be a "rolling window" or
            queue. We remove the oldest loss value. You may keep track of the full history of validation loss values
            during training, but maintain a separate list in `fit()` for this.

        Conditions that determine whether to stop training early:
        3. We are only eligable for aborting training when the rolling window is exactly `patience`+1 (after enforcing
        length above).
        4. We stop early when the OLDEST rolling validation loss (`curr_val_loss`) is as small as or smaller than the
        other `patience` more recent validation loss values in the window.

        Parameters:
        -----------
        recent_val_losses: Python list of floats. len between 0 and `patience`+1 (inclusive).
            Recently computed losses on the validation set.
        curr_val_loss: float
            The loss computed on the validation set on the current training epoch.
        patience: int.
            The patience: how many recent loss values computed on the validation set we should consider when deciding
            whether to stop training early.

        Returns:
        -----------
        recent_val_losses: Python list of floats. len between 1 and `patience`+1 (inclusive).
            The list of recent validation loss values passsed into this method updated to include the current validation
            loss.
        stop. bool.
            Should we stop training based on the recent validation loss values and the patience value?

        NOTE:
        - This method can be concisely implemented entirely with regular Python (TensorFlow/Numpy not needed).
        - It may be helpful to think of `recent_val_losses` as a queue: the current loss value always gets inserted
        either at the beginning or end. The oldest value is then always on the other end of the list.
        '''
        # Always append the current validation loss
        recent_val_losses.append(curr_val_loss)
        
        # Enforce maximum window size of patience + 1
        if len(recent_val_losses) > patience + 1:
            recent_val_losses.pop(0)  # Remove oldest (first) element
        
        # Check if we should stop early
        stop = False
        if len(recent_val_losses) == patience + 1:
            # Window is full, check if oldest loss is <= all other losses
            oldest_loss = recent_val_losses[0]
            other_losses = recent_val_losses[1:]
            if oldest_loss <= min(other_losses):
                stop = True
        
        return recent_val_losses, stop

    def lr_step_decay(self, lr_decay_rate):
        '''Adjusts the learning rate used by the optimizer to be a proportion `lr_decay_rate` of the current learning
        rate.

        (Week 3)

        Paramters:
        ----------
        lr_decay_rate: float.
            A value between 0.0. and 1.0 that represents the proportion of the current learning rate that the learning
            rate should be set to. For example, 0.7 would mean the learning rate is set to 70% its previous value.

        NOTE: TensorFlow optimizer objects store the learning rate as a field called learning_rate.
        Example: self.opt.learning_rate for the optimizer object named self.opt. You are allowed to modify it with
        regular Python assignment.

        TODO:
        1. Update the optimizer's learning rate.
        2. Always print out the optimizer's learning rate before and after the change.
        '''
        # Print current learning rate
        current_lr = float(self.opt.learning_rate)
        print(f'Learning rate before decay: {current_lr}')
        
        # Update learning rate
        new_lr = current_lr * lr_decay_rate
        self.opt.learning_rate = new_lr
        
        # Print new learning rate
        print(f'Learning rate after decay: {float(self.opt.learning_rate)}')
