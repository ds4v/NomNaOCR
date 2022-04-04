import tensorflow as tf


class EarlyStoppingWithStuck(tf.keras.callbacks.Callback):
    def __init__(self, patience=0, stuck_str=None):
        super(EarlyStoppingWithStuck, self).__init__()
        self.patience = patience
        self.stuck_str = stuck_str
        self.best_weights = None

    def on_train_begin(self, logs=None):
        self.wait = 0 # Number of epoch it has waited when loss is no longer minimum
        self.stopped_epoch = 0 # The epoch the training stops at
        self.best_loss = float('inf') # Initialize the best loss as infinity
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        loss, val_loss = logs.get('loss'), logs.get('val_loss')
        if tf.less(val_loss, self.best_loss) and not eval(self.stuck_str):
            self.wait = 0
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print(f'Early stopping and restored the model weights from the end of ' 
                  f'epoch {self.best_epoch + 1} - val_loss: {self.best_loss}\n')
