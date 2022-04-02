import tensorflow as tf
from utils import ctc_decode


class MeanEditDistance(tf.keras.metrics.Metric):
    def __init__(
        self, 
        padding_token = 0,
        max_length = 0, # Required if use_ctc_decode == True
        use_ctc_decode = False, # Need to decode predictions if CTC loss used,
        name = 'edist', 
        **kwargs
    ):
        super(MeanEditDistance, self).__init__(name=name, **kwargs)
        self.epoch_edit_distance = self.add_weight(name='med', initializer='zeros')
        self.padding_token = padding_token
        self.use_ctc_decode = use_ctc_decode
        self.max_length = max_length

    # https://github.com/solivr/tf-crnn/blob/master/tf_crnn/model.py#L157
    def _calculate_batch_edist(self, y_true, y_pred):
        # Get a single batch and convert its labels to sparse tensors.
        idxs = tf.where(y_true != self.padding_token)
        sparse_true = tf.SparseTensor(
            tf.cast(idxs, tf.int64),
            tf.gather_nd(y_true, idxs),
            tf.cast(tf.shape(y_true), tf.int64)
        )

        idxs = tf.where(tf.logical_and(
            y_true != self.padding_token, 
            y_true != -1 # For blank labels if use_ctc_decode 
        ))
        sparse_pred = tf.SparseTensor(
            tf.cast(idxs, tf.int64),
            tf.gather_nd(y_pred, idxs),
            tf.cast(tf.shape(y_pred), tf.int64)
        )

        # Compute individual edit distances and average them out.
        # https://stackoverflow.com/questions/51612489
        edit_distances = tf.edit_distance(sparse_pred, sparse_true, normalize=False)
        return tf.reduce_mean(edit_distances)

    def update_state(self, y_true, y_pred, **kwargs):
        if self.use_ctc_decode: y_pred = ctc_decode(y_pred, self.max_length)
        self.epoch_edit_distance.assign(tf.reduce_mean([
            self.epoch_edit_distance, 
            self._calculate_batch_edist(y_true, y_pred)
        ]))
    
    def result(self):
        return self.epoch_edit_distance

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.epoch_edit_distance.assign(0.0)
