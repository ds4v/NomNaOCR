import tensorflow as tf


class EarlyStoppingWithStuck(tf.keras.callbacks.Callback):
    def __init__(self, edist_metric_name='edist', patience=0, min_delta=0, stuck_str=None):
        super(EarlyStoppingWithStuck, self).__init__()
        self.edist_metric_name = edist_metric_name
        self.patience = patience
        self.min_delta = min_delta
        self.stuck_str = stuck_str
        self.best_weights = None

    def on_train_begin(self, logs=None):
        self.wait = 0 # Number of epoch it has waited when loss is no longer minimum
        self.stopped_epoch = 0 # The epoch the training stops at
        self.best_loss = float('inf') # Initialize the best loss as infinity
        self.best_edist = float('inf') # Initialize the best mean edit distance as infinity
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get('loss')
        edist = logs.get(self.edist_metric_name)

        val_loss = logs.get('val_loss')
        val_edist = logs.get(f'val_{self.edist_metric_name}')

        if (self.best_loss - val_loss > self.min_delta or 
            self.best_edist - val_edist > self.min_delta) and \
            not eval(self.stuck_str):
            self.wait = 0
            self.best_loss = val_loss
            self.best_edist = val_edist
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
            print(f'Early stopping and restored weights from epoch {self.best_epoch + 1}' 
                  f' - val_loss: {self.best_loss:.4f} - val_edist: {self.best_edist:.4f}\n')
