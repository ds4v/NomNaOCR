import numpy as np
import tensorflow as tf


class CTCLayer(tf.keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super().__init__(name='ctc_loss', **kwargs)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred, label_length):
        batch_length = tf.cast(tf.shape(y_true)[0], dtype='int64')
        input_length = tf.cast(tf.shape(y_pred)[1], dtype='int64')
        input_length *= tf.ones(shape=(batch_length, 1), dtype='int64')

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        return y_pred  # At test time, just return the computed predictions.
        
        
# A utility function to decode the output of the network
def decode_batch_predictions(preds, max_length, beam_width, num2char_func):
    # Use greedy search. For complex tasks, you can use beam search
    preds_decoded = tf.keras.backend.ctc_decode(
        preds,
        input_length = np.ones(preds.shape[0]) * preds.shape[1],
        greedy = False,
        beam_width = beam_width
    )[0][0][:, :max_length]

    # Iterate over the results and get back the text
    output_text = []
    for result in preds_decoded:
        result = tf.gather(result, tf.where(tf.math.not_equal(result, -1)))
        result = tf.strings.reduce_join(num2char_func(result))
        output_text.append(result.numpy().decode('utf-8'))
    return output_text