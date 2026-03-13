'''extension_jacob.py
Extension upon base project DeepNetwork and Decoder classes.
Jacob Petty, Sardor Nodirov, and Saad Khan
CS 443: Bio-Inspired Learning
'''
import time
import numpy as np
import tensorflow as tf

from network import DeepNetwork
from layers import Dense

class DeepNetwork_Extended(DeepNetwork):
    def __init__(self, input_feats_shape):
        '''Basically inherits everything from parent without change.'''
        super().__init__(input_feats_shape)

    def fit(self, x, y, x_val=None, y_val=None, batch_size=128, max_epochs=10000, val_every=1,
            verbose=True, patience=999, lr_patience=999, lr_decay_amount=0.00005, lr_max_decays=12):
        '''
        ONLY 1 Change from old fit method params:
        
        lr_decay_amount: float.
            A value between 0.0. and 1.0 that represents the amount to be taken off the current learning rate that the learning
            rate should be set to. Different from fractional multiplicative decay from before.

        '''
        train_loss_hist = []
        val_loss_hist = []
        val_acc_hist = []

        early_stopping_recent_val_losses = [] 
        lr_decay_recent_rolling_val_losses = []

        num_decays = 0

        N = int(x.shape[0])
        num_batches = N // batch_size
        if num_batches < 1:
            num_batches = 1

        self.set_layer_training_mode(is_training=True)

        total_start = time.time()

        rng = np.random.default_rng(0)

        # PART OF EXTENSION
        decayed_max_amount = False # Catch so we don't decay lr more than possible. Basically a stop that could get triggered before lr_max_decays

        for e in range(max_epochs):
            epoch_start = time.time()

            epoch_loss = 0
            for b in range(num_batches):
                batch_indices = rng.integers(low=0, high=N, size=batch_size)
                x_batch = tf.gather(x, batch_indices)
                y_batch = tf.gather(y, batch_indices)

                loss = self.train_step(x_batch, y_batch)
                epoch_loss += loss

            avg_epoch_loss = epoch_loss / num_batches
            train_loss_hist.append(float(avg_epoch_loss))

            epoch_time = time.time() - epoch_start

            val_acc = None
            val_loss = None

            if x_val is not None and y_val is not None and (e + 1) % val_every == 0:
                val_acc, val_loss = self.evaluate(x_val, y_val)
                val_loss_hist.append(float(val_loss))
                val_acc_hist.append(float(val_acc))
                self.set_layer_training_mode(is_training=True)
                
                early_stopping_recent_val_losses, should_stop = self.early_stopping(early_stopping_recent_val_losses, val_loss, patience)

                if should_stop:
                    if verbose:
                        print(f'Epoch {e+1}/{max_epochs}, Training loss {float(avg_epoch_loss):.3f}, Val loss {val_loss:.3f}, Val acc {val_acc:.4f}. Epoch took: {epoch_time:.1f} secs')
                        print(f'Early stopping triggered at epoch {e+1}')
                    break
                
                lr_decay_recent_rolling_val_losses, should_decay = self.early_stopping(lr_decay_recent_rolling_val_losses, val_loss, lr_patience)

                # PART OF EXTENSION
                if should_decay and num_decays < lr_max_decays and decayed_max_amount == False:

                    # Get old lr
                    old = float(self.opt.learning_rate)

                    # Decay
                    self.lr_step_decay(lr_decay_amount, verbose=verbose)

                    # Get new lr
                    new = float(self.opt.learning_rate)

                    # Make sure we actually did decayed. If so, add one to total, but if not, make sure we don't anymore
                    if old != new:
                        num_decays += 1
                    else:
                        decayed_max_amount = True

                    if num_decays == lr_max_decays and verbose:
                        print("Hit maximum number of lr decays!")

            if verbose:
                val_acc_str = f'{val_acc:.4f}' if val_acc is not None else 'N/A'
                val_loss_str = f'{val_loss:.3f}' if val_loss is not None else 'N/A'
                print(f'Epoch {e+1}/{max_epochs}, Training loss {float(avg_epoch_loss):.3f}, Val loss {val_loss_str}, Val acc {val_acc_str}. Epoch took: {epoch_time:.1f} secs')

        total_time = time.time() - total_start

        if verbose:
            print(f'Training finished after {e+1} epochs in {total_time:.2f} seconds')
            print(f'The learning rate was decayed {num_decays} times.')
            
        return train_loss_hist, val_loss_hist, val_acc_hist, e


    def lr_step_decay(self, lr_decay_amount, verbose):
        '''Adjusts the learning rate used by the optimizer by subtracting a fixed portion off the current learning 
        rate. 

        Paramters:
        ----------
        lr_decay_amount: float.
            A value between 0.0. and 1.0 that represents the change to be made to the current learning rate that the learning
            rate should be set to. 
        '''
        current_lr = float(self.opt.learning_rate)

        if current_lr - lr_decay_amount <= 0:
            if verbose:
                print("Cannot decay learning rate any further! Learning rate not changed!")
        else:
            new_lr = current_lr - lr_decay_amount 
            self.opt.learning_rate = new_lr
            if verbose:
                print(f'Learning rate before decay: {current_lr} --> Learning rate after decay: {float(self.opt.learning_rate)}')

class LinearDecoder_Extended(DeepNetwork_Extended):
    def __init__(self, input_feats_shape, C):
        super().__init__(input_feats_shape)
        self.output_layer = Dense('Output Layer', C, activation='softmax', prev_layer_or_block=None)
        

    def __call__(self, x):
        return self.output_layer(x)

class NonlinearDecoder_Extended(DeepNetwork_Extended):
    def __init__(self, input_feats_shape, C, wt_scale=0.1, beta=0.0025, loss_exp=2.0):
        super().__init__(input_feats_shape)
        self.output_layer = Dense('Output Layer', C, activation='tanh', 
                                   prev_layer_or_block=None, wt_scale=wt_scale)
        self.output_layer.set_tanh_beta(beta)
        self.output_layer.loss_exp = loss_exp

    def __call__(self, x):
        return self.output_layer(x)
