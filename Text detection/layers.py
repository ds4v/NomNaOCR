import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, ReLU


class ConvBnRelu(tf.keras.layers.Layer):
    def __init__(self, filters=64, kernel_size=3, use_bias=True, name='ConvBnRelu', **kwargs):
        super(ConvBnRelu, self).__init__(name=name, **kwargs)
        self.conv = Conv2D(filters, kernel_size, padding='same', use_bias=use_bias)
        self.bn = BatchNormalization()
        self.relu = ReLU()
        
    def call(self, inputs, training):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        return self.relu(x)


class DeConvMap(tf.keras.layers.Layer):
    def __init__(self, filters=64, name='DeConvMap', **kwargs):
        super(DeConvMap, self).__init__(name=name, **kwargs)
        self.conv_bn = ConvBnRelu(filters, kernel_size=3, use_bias=False)
        self.deconv1 = Conv2DTranspose(filters, kernel_size=2, strides=2, use_bias=False)
        self.bn = BatchNormalization()
        self.relu = ReLU()
        self.deconv2 = Conv2DTranspose(1, kernel_size=2, strides=2, activation='sigmoid')
        
    def call(self, inputs, training):
        x = self.conv_bn(inputs)
        x = self.deconv1(x)
        x = self.bn(x, training=training)
        x = self.relu(x)
        x = self.deconv2(x)
        return tf.squeeze(x, axis=-1)
