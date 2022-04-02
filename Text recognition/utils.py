import tensorflow as tf


def ctc_decode(predictions, max_length):
    input_length = tf.ones(len(predictions)) * predictions.shape[1]
    preds_decoded = tf.keras.backend.ctc_decode(
        predictions,
        input_length = input_length,
        greedy = True,
    )[0][0][:, :max_length]
    return preds_decoded


def decode_batch_predictions(preds, max_length, num2char_func):
    preds_decoded = ctc_decode(preds, max_length)
    output_text = []

    # Iterate over the results and get back the text
    for result in preds_decoded:
        result = tf.gather(result, tf.where(result != -1))
        result = tf.strings.reduce_join(num2char_func(result))
        output_text.append(result.numpy().decode('utf-8'))
    return output_text
