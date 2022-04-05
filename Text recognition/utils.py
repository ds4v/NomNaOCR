import tensorflow as tf


def ctc_decode(predictions, max_length):
    input_length = tf.ones(len(predictions)) * predictions.shape[1]
    preds_decoded = tf.keras.backend.ctc_decode(
        predictions,
        input_length = input_length,
        greedy = True,
    )[0][0][:, :max_length]
    return preds_decoded