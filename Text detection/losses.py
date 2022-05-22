import tensorflow as tf
import tensorflow.keras.backend as K


def balanced_crossentropy_loss(pred, gt, mask, negative_ratio=3.):
    pred = pred[..., 0]
    positive_mask = (gt * mask)
    negative_mask = ((1 - gt) * mask)
    positive_count = tf.reduce_sum(positive_mask)
    negative_count = tf.reduce_min([tf.reduce_sum(negative_mask), positive_count * negative_ratio])

    loss = K.binary_crossentropy(gt, pred)
    positive_loss = loss * positive_mask
    negative_loss = loss * negative_mask
    
    negative_loss, _ = tf.nn.top_k(tf.reshape(negative_loss, (-1,)), tf.cast(negative_count, tf.int32))
    balanced_loss = (tf.reduce_sum(positive_loss) + tf.reduce_sum(negative_loss)) / (positive_count + negative_count + 1e-6)
    return balanced_loss, loss


def dice_loss(pred, gt, mask, weights):
    pred = pred[..., 0]
    weights = (weights - tf.reduce_min(weights)) / (tf.reduce_max(weights) - tf.reduce_min(weights) + 1e-6) + 1.
    mask = mask * weights
    intersection = tf.reduce_sum(pred * gt * mask)
    union = tf.reduce_sum(pred * mask) + tf.reduce_sum(gt * mask) + 1e-6
    return 1 - 2.0 * intersection / union


def l1_loss(pred, gt, mask):
    pred = pred[..., 0]
    mask_sum = tf.reduce_sum(mask)
    return K.switch(mask_sum > 0, tf.reduce_sum(tf.abs(pred - gt) * mask) / (mask_sum + 1e-6), 0.)


def db_loss(args, alpha=5.0, beta=10.0, ohem_ratio=3.0):
    gt_input, mask_input, thresh_input, thresh_mask_input, binarize_map, thresh_binary, threshold_map = args
    threshold_loss = l1_loss(threshold_map, thresh_input, thresh_mask_input)
    binarize_loss, dice_loss_weights = balanced_crossentropy_loss(binarize_map, gt_input, mask_input, negative_ratio=ohem_ratio)
    thresh_binary_loss = dice_loss(thresh_binary, gt_input, mask_input, dice_loss_weights)
    return alpha * binarize_loss + beta * threshold_loss + thresh_binary_loss
