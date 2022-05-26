import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy


class DBLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=5.0, beta=10.0, ohem_ratio=3.0, name='DBLoss', **kwargs):
        super(DBLoss, self).__init__(name=name, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.ohem_ratio = ohem_ratio
        self.bce_loss_func = BinaryCrossentropy()


    def balanced_bce_loss(self, pred, gt, mask, negative_ratio=3.):
        positive_mask = (gt * mask)
        negative_mask = ((1 - gt) * mask)
        positive_count = tf.reduce_sum(positive_mask)
        negative_count = tf.reduce_min([tf.reduce_sum(negative_mask), positive_count * negative_ratio])

        bce_loss = self.bce_loss_func(gt, pred)
        positive_loss = bce_loss * positive_mask
        negative_loss = bce_loss * negative_mask
        
        negative_loss, _ = tf.nn.top_k(tf.reshape(negative_loss, (-1,)), tf.cast(negative_count, tf.int32))
        sum_losses = tf.reduce_sum(positive_loss) + tf.reduce_sum(negative_loss)
        balanced_loss = sum_losses / (positive_count + negative_count + 1e-6)
        return balanced_loss, bce_loss


    def dice_loss(self, pred, gt, mask, weights):
        weights = (weights - tf.reduce_min(weights)) / (tf.reduce_max(weights) - tf.reduce_min(weights) + 1e-6) + 1.
        mask *= weights
        intersection = tf.reduce_sum(pred * gt * mask)
        union = tf.reduce_sum(pred * mask) + tf.reduce_sum(gt * mask) + 1e-8
        return 1 - 2.0 * intersection / union


    def l1_loss(self, pred, gt, mask):
        mask_sum = tf.reduce_sum(mask)
        return tf.where(mask_sum > 0, tf.reduce_sum(tf.abs(pred - gt) * mask) / (mask_sum + 1e-6), 0.)


    def call(self, y_true, y_pred):
        gt, mask, thresh_map, thresh_mask = y_true
        binarize_map, threshold_map, thresh_binary = y_pred
        Lt = self.l1_loss(threshold_map, thresh_map, thresh_mask)
        Lb, dice_loss_weights = self.balanced_bce_loss(binarize_map, gt, mask, negative_ratio=self.ohem_ratio)
        Ls = self.dice_loss(thresh_binary, gt, mask, dice_loss_weights)
        return self.alpha * Lb + self.beta * Lt + Ls # L = Ls + α × Lb + β × Lt 