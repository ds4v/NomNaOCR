import numpy as np
import tensorflow as tf
from ctc import ctc_decode


class EditDistanceCallback(tf.keras.callbacks.Callback):
    def __init__(
        self, 
        prediction_model, 
        valid_tf_dataset, 
        max_length, 
        use_ctc_decode = False, # Need to decode predictions if CTC loss used
        beam_width = 20 # Required if use_ctc_decode == True
    ):
        super(EditDistanceCallback, self).__init__()
        self.prediction_model = prediction_model
        self.max_length = max_length
        self.use_ctc_decode = use_ctc_decode
        self.beam_width = beam_width
        self.logs = []

        self.images, self.labels = [], []
        for batch in valid_tf_dataset: 
            self.images.append(batch['image'])
            self.labels.append(batch['label'])

    def _calculate_edit_distance(self, labels, preds):
        # Get a single batch and convert its labels to sparse tensors.
        sparse_labels = tf.cast(tf.sparse.from_dense(labels), dtype='int64')
        sparse_preds = tf.cast(tf.sparse.from_dense(preds), dtype='int64')

        # Compute individual edit distances and average them out.
        # https://stackoverflow.com/questions/51612489
        edit_distances = tf.edit_distance(sparse_preds, sparse_labels, normalize=False)
        return tf.reduce_mean(edit_distances).numpy()

    def on_epoch_end(self, epoch, logs=None):
        edit_distances = []
        for batch_images, batch_labels in zip(self.images, self.labels):
            preds = self.prediction_model.predict(batch_images)
            if self.use_ctc_decode: 
                preds = ctc_decode(preds, self.max_length, self.beam_width)
            edit_distances.append(self._calculate_edit_distance(batch_labels, preds))

        mean_edit_distances = np.mean(edit_distances)
        self.logs.append(mean_edit_distances)
        print(' - Mean edit distance:', mean_edit_distances)


class EarlyStoppingWithStuck(tf.keras.callbacks.Callback):
    def __init__(self, patience=0, stuck_str=None):
        super(EarlyStoppingWithStuck, self).__init__()
        self.patience = patience
        self.stuck_str = stuck_str
        self.best_weights = None

    def on_train_begin(self, logs=None):
        self.wait = 0 # Number of epoch it has waited when loss is no longer minimum
        self.stopped_epoch = 0 # The epoch the training stops at
        self.best_loss = np.Inf # Initialize the best loss as infinity
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        loss, val_loss = logs.get('loss'), logs.get('val_loss')
        if np.less(val_loss, self.best_loss) and not eval(self.stuck_str):
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
