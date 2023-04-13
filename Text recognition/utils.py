import csv
import tensorflow as tf
from tqdm import tqdm


def ctc_decode(predictions, max_length):
    input_length = tf.ones(len(predictions)) * predictions.shape[1]
    preds_decoded = tf.keras.backend.ctc_decode(
        predictions,
        input_length = input_length,
        greedy = True,
    )[0][0][:, :max_length]
    
    return tf.where(
        preds_decoded == tf.cast(1, tf.int64),
        tf.cast(-1, tf.int64), # Treat [UNK] token same as blank label
        preds_decoded
    )


def update_tensor_column(tensor, values, col_idx):
    if col_idx < 0: raise ValueError("col_idx must be >= 0")
    rows = tf.range(tf.shape(tensor)[0])
    column = tf.zeros_like(rows) + col_idx
    idxs = tf.stack([rows, column], axis=1)
    return tf.tensor_scatter_nd_update(tensor, idxs, tf.squeeze(values, axis=-1))


def tokens2sparse(batch_tokens):
    idxs = tf.where(tf.logical_and(
        batch_tokens != 0, # For [PAD] token
        batch_tokens != -1 # For blank label if use_ctc_decode 
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


def rec2csv(file_name, patch_list, data_handler, model, use_ctc_decode=False):
    with open(file_name, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        for img_path in tqdm(patch_list):
            image = data_handler.process_image(img_path)
            pred_tokens = model.predict(tf.expand_dims(image, axis=0))
            pred_labels = data_handler.tokens2texts(pred_tokens, use_ctc_decode)
            writer.writerow([img_path, pred_labels[0]])
