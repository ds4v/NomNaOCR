# https://github.com/MhLiao/DB/blob/master/decoders/seg_detector.py
import tensorflow as tf
from tensorflow.keras.layers import Input, UpSampling2D, Add, Concatenate, Lambda
from layers import ConvBnRelu, DeConvMap
from keras_resnet.models import *


class DBNet(tf.keras.Model):
    def __init__(self, post_processor, backbone='ResNet50', k=50, name='DBNet', **kwargs):
        super().__init__()
        self.model = self._build_model(backbone, k, name)
        self.post_processor = post_processor
        

    def _build_model(self, backbone='ResNet50', k=50, name='DBNet'):
        image_input = Input(shape=(None, None, 3), name='image')
        backbone = eval(f'{backbone}(inputs=image_input, include_top=False)')
        
        C2, C3, C4, C5 = backbone.outputs
        in2 = ConvBnRelu(256, 1, name='in2')(C2)
        in3 = ConvBnRelu(256, 1, name='in3')(C3)
        in4 = ConvBnRelu(256, 1, name='in4')(C4)
        in5 = ConvBnRelu(256, 1, name='in5')(C5)
        
        # The pyramid features are up-sampled to the same scale and cascaded to produce feature F
        out4 = UpSampling2D(2, name='up5')(in5)  + in4
        out3 = UpSampling2D(2, name='up4')(out4) + in3
        out2 = UpSampling2D(2, name='up3')(out3) + in2
        
        P5 = tf.keras.Sequential([ConvBnRelu(64, 3), UpSampling2D(8)], name='P5')(in5)
        P4 = tf.keras.Sequential([ConvBnRelu(64, 3), UpSampling2D(4)], name='P4')(out4)
        P3 = tf.keras.Sequential([ConvBnRelu(64, 3), UpSampling2D(2)], name='P3')(out3)
        P2 = ConvBnRelu(64, 3, name='P2')(out2)
        
        # Calculate DBNet maps
        fuse = Concatenate(name='fuse')([P2, P3, P4, P5]) # (batch_size, /4, /4, 256)
        binarize_map = DeConvMap(64, name='probability_map')(fuse)
        threshold_map = DeConvMap(64, name='threshold_map')(fuse)
        thresh_binary = Lambda( 
            lambda x: 1 / (1 + tf.exp(-k * (x[0] - x[1]))), # b_hat = 1 / (1 + e^(-k(P - T)))
            name = 'approximate_binary_map'
        )([binarize_map, threshold_map]) 
        
        return tf.keras.Model(
            inputs = image_input, 
            outputs = [binarize_map, threshold_map, thresh_binary], 
            name = name
        )


    @tf.function
    def train_step(self, batch):
        with tf.GradientTape() as tape:
            y_pred = self.model(batch[0], training=True)
            loss = self.loss(batch[-1], y_pred)

        # Apply an optimization step
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {'loss': loss}


    @tf.function
    def test_step(self, batch):
        y_pred = self.model(batch[0], training=False)
        loss = self.loss(batch[-1], y_pred)
        return {'loss': loss}


    def predict(self, batch_images, batch_true_sizes, output_polygon=False):
        binarize_map, _, _ = self.model(batch_images, training=False)
        batch_boxes, batch_scores = self.post_processor(
            binarize_map = binarize_map.numpy(), 
            batch_true_sizes = batch_true_sizes, 
            output_polygon = output_polygon
        )
        return batch_boxes, batch_scores
