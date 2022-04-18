import tensorflow as tf


def ctc_decode(predictions, max_length):
    input_length = tf.ones(len(predictions)) * predictions.shape[1]
    preds_decoded = tf.keras.backend.ctc_decode(
        predictions,
        input_length = input_length,
        greedy = True,
    )[0][0][:, :max_length]
    return preds_decoded


def tokens2sparse(batch_tokens, padding_token):
    idxs = tf.where(tf.logical_and(
        batch_tokens != padding_token, 
        batch_tokens != -1 # For blank labels if use_ctc_decode 
    ))
    return tf.SparseTensor(
        tf.cast(idxs, tf.int64),
        tf.gather_nd(batch_tokens, idxs),
        tf.cast(tf.shape(batch_tokens), tf.int64)
    )


def sparse2dense(tensor, shape):
    tensor = tf.sparse.reset_shape(tensor, shape)
    tensor = tf.sparse.to_dense(tensor, default_value=-1)
    tensor = tf.cast(tensor, tf.float32)
    return tensor
