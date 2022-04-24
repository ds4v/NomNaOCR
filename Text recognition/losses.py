import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses import SparseCategoricalCrossentropy


class CTCLoss(tf.keras.losses.Loss):
    def __init__(self, padding_token=0, name='ctc_loss', **kwargs):
        super(CTCLoss, self).__init__(name=name, **kwargs)
        self.padding_token = padding_token

    def call(self, y_true, y_pred):
        label_length = tf.cast(y_true != self.padding_token, tf.int64)
        label_length = tf.expand_dims(tf.reduce_sum(label_length, axis=-1), axis=1)

        batch_length = tf.cast(tf.shape(y_true)[0], tf.int64)
        pred_length = tf.cast(tf.shape(y_pred)[1], tf.int64)
        pred_length *= tf.ones((batch_length, 1), tf.int64)
        return K.ctc_batch_cost(y_true, y_pred, pred_length, label_length)


class MaskedLoss(tf.keras.losses.Loss):
    def __init__(self, padding_token=0, name='masked_loss', **kwargs):
        super(MaskedLoss, self).__init__(name=name, **kwargs)
        # The padding_token need to be 0 for SparseCategoricalCrossentropy
        # See https://stackoverflow.com/questions/63171001 if loss == nan
        self.loss = SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        self.padding_token = padding_token

    def call(self, y_true, y_pred):
        loss = self.loss(y_true, y_pred)
        mask = tf.cast(y_true != self.padding_token, tf.float32)
        return tf.math.divide_no_nan(
            tf.reduce_sum(loss * mask), 
            tf.reduce_sum(mask) # Actual sequence length
        )
