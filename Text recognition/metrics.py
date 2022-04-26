# https://github.com/FLming/CRNN.tf2/blob/master/crnn/metrics.py
import tensorflow as tf
from utils import ctc_decode, tokens2sparse, sparse2dense


class SequenceAccuracy(tf.keras.metrics.Metric):
    def __init__(
        self, 
        use_ctc_decode = False, # Need to decode predictions if CTC loss used,
        name = 'seq_acc', 
        **kwargs
    ):
        super(SequenceAccuracy, self).__init__(name=name, **kwargs)
        self.use_ctc_decode = use_ctc_decode
        self.count = self.add_weight(name='count', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')
                
    def update_state(self, y_true, y_pred, **kwargs):
        batch_size, max_length = tf.shape(y_true)[0], tf.shape(y_true)[1]
        if self.use_ctc_decode: y_pred = ctc_decode(y_pred, max_length)

        # Get a single batch and convert its labels to sparse tensors.
        sparse_true = tokens2sparse(y_true)
        sparse_pred = tokens2sparse(y_pred)

        y_true = sparse2dense(sparse_true, [batch_size, max_length])
        y_pred = sparse2dense(sparse_pred, [batch_size, max_length])

        num_errors = tf.reduce_any(y_true != y_pred, axis=1)
        num_errors = tf.reduce_sum(tf.cast(num_errors, tf.float32))
        total = tf.cast(batch_size, tf.float32)

        self.count.assign_add(total - num_errors)
        self.total.assign_add(total)

    def result(self):
        return tf.math.divide_no_nan(self.count, self.total)

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.count.assign(0)
        self.total.assign(0)


class CharacterAccuracy(tf.keras.metrics.Metric):
    def __init__(
        self, 
        use_ctc_decode = False, # Need to decode predictions if CTC loss used,
        name = 'char_acc', 
        **kwargs
    ):
        super(CharacterAccuracy, self).__init__(name=name, **kwargs)
        self.use_ctc_decode = use_ctc_decode
        self.count = self.add_weight(name='count', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')
                
    def update_state(self, y_true, y_pred, **kwargs):
        batch_size, max_length = tf.shape(y_true)[0], tf.shape(y_true)[1]
        if self.use_ctc_decode: y_pred = ctc_decode(y_pred, max_length)

        num_errors = tf.logical_and(y_true != y_pred, y_true != 0)
        num_errors = tf.reduce_sum(tf.cast(num_errors, tf.float32))
        total = tf.reduce_sum(tf.cast(y_true != 0, tf.float32))

        self.count.assign_add(total - num_errors)
        self.total.assign_add(total)

    def result(self):
        return tf.math.divide_no_nan(self.count, self.total)

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.count.assign(0)
        self.total.assign(0)


class LevenshteinDistance(tf.keras.metrics.Metric):
    def __init__(
        self, 
        use_ctc_decode = False, # Need to decode predictions if CTC loss used,
        normalize = False, # If True, this becomes Character Error Rate: CER = (S + D + I) / N
        name = 'levenshtein_distance', 
        **kwargs
    ):
        super(LevenshteinDistance, self).__init__(name=name, **kwargs)
        self.use_ctc_decode = use_ctc_decode
        self.normalize = normalize
        self.sum_distance = self.add_weight(name='sum_distance', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')

    def update_state(self, y_true, y_pred, **kwargs):
        batch_size, max_length = tf.shape(y_true)[0], tf.shape(y_true)[1]
        if self.use_ctc_decode: y_pred = ctc_decode(y_pred, max_length)

        # Get a single batch and convert its labels to sparse tensors.
        sparse_true = tokens2sparse(y_true)
        sparse_pred = tokens2sparse(y_pred)

        # Explain tf.edit_distance: https://stackoverflow.com/questions/51612489
        edit_distances = tf.edit_distance(sparse_pred, sparse_true, normalize=self.normalize)
        self.sum_distance.assign_add(tf.reduce_sum(edit_distances))
        self.total.assign_add(tf.cast(batch_size, tf.float32))
    
    def result(self):
        # Computes and returns a scalar value for the metric
        return tf.math.divide_no_nan(self.sum_distance, self.total)

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.sum_distance.assign(0)
        self.total.assign(0)


# https://github.com/solivr/tf-crnn/blob/master/tf_crnn/model.py#L157
# The result of this function is the same as that of the 
# LevenshteinDistance metric above with normalize = True
def warp_cer_metric(y_true, y_pred, use_ctc_decode=False):
    ''' How to use:
    from tensorflow.keras.metrics import MeanMetricWrapper
    cer = MeanMetricWrapper(lambda y_true, y_pred: warp_cer_metric(
        y_true, y_pred, use_ctc_decode=True
    ), name='cer')
    '''
    if use_ctc_decode: y_pred = ctc_decode(y_pred, tf.shape(y_true)[1])
    y_true = tf.cast(y_true, tf.int64)

    # Get a single batch and convert its labels to sparse tensors.
    sparse_true = tokens2sparse(y_true)
    sparse_pred = tokens2sparse(y_pred)

    # Explain tf.edit_distance: https://stackoverflow.com/questions/51612489
    edit_distances = tf.edit_distance(sparse_pred, sparse_true, normalize=False)

    # Compute edit distance and total chars count
    sum_distance = tf.reduce_sum(edit_distances)
    count_chars = tf.reduce_sum(tf.cast(y_true != 0, tf.float32))
    return tf.math.divide_no_nan(sum_distance, count_chars, name='cer')