import numpy as np
import tensorflow as tf


class EditDistanceCallback(tf.keras.callbacks.Callback):
    def __init__(self, prediction_model, tf_dataset, max_length):
        self.prediction_model = prediction_model
        self.max_length = max_length
        self.logs = []

        self.images, self.labels = [], []
        for batch in tf_dataset:
            self.images.append(batch['image'])
            self.labels.append(batch['label'])
        


    def _calculate_edit_distance(self, labels, predictions):
        # Make predictions and convert them to sparse tensors.
        preds_decoded = tf.keras.backend.ctc_decode(
            predictions,
            input_length = np.ones(predictions.shape[0]) * predictions.shape[1],
            greedy = True
        )[0][0][:, :self.max_length]

        # Get a single batch and convert its labels to sparse tensors.
        sparse_labels = tf.cast(tf.sparse.from_dense(labels), dtype='int64')
        sparse_preds = tf.cast(tf.sparse.from_dense(preds_decoded), dtype='int64')

        # Compute individual edit distances and average them out.
        edit_distances = tf.edit_distance(sparse_preds, sparse_labels, normalize=False)
        return tf.reduce_mean(edit_distances).numpy()


    def on_epoch_end(self, epoch, logs=None):
        edit_distances = []
        for i in range(len(self.images)):
            preds = self.prediction_model.predict(self.images[i])
            edit_distances.append(self._calculate_edit_distance(self.labels[i], preds))

        mean_edit_distances = np.mean(edit_distances)
        self.logs.append(mean_edit_distances)
        print(' - Mean edit distance:', mean_edit_distances)
