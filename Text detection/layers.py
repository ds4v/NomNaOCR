import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Activation


class ConvBnRelu(tf.keras.layers.Layer):
    def __init__(self, filters=64, kernel_size=3, use_bias=True, name='ConvBnRelu', **kwargs):
        super(ConvBnRelu, self).__init__(name=name, **kwargs)
        self.conv = Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal', use_bias=use_bias)
        self.bn = BatchNormalization()
        self.relu = Activation('relu')
        
    def call(self, inputs, training):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        return self.relu(x)


class DeconvolutionalMap(tf.keras.layers.Layer):
    def __init__(self, filters=64, name='DeconvolutionalMap', **kwargs):
        super(DeconvolutionalMap, self).__init__(name=name, **kwargs)
        self.conv_bn = ConvBnRelu(filters, 3, use_bias=False)
        self.deconv1 = Conv2DTranspose(filters, 2, strides=2, kernel_initializer='he_normal', use_bias=False)
        self.bn2 = BatchNormalization()
        self.relu2 = Activation('relu')
        self.deconv2 = Conv2DTranspose(1, 2, strides=2, kernel_initializer='he_normal', activation='sigmoid')
        
    def call(self, inputs, training):
        x = self.conv_bn(inputs)
        x = self.deconv1(x)
        x = self.bn2(x, training=training)
        x = self.relu2(x)
        x = self.deconv2(x)
        return tf.squeeze(x, axis=-1)