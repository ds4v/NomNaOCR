import tensorflow as tf
from tensorflow.keras.layers import Input, UpSampling2D, Add, Concatenate, Lambda
from layers import ConvBnRelu, DeconvolutionalMap
from keras_resnet.models import ResNet50
from abc import abstractmethod, ABCMeta # For define pure virtual functions


class CustomTrainingModel(tf.keras.Model, metaclass=ABCMeta):
    def __init__(self, name='CustomTrainingModel', **kwargs):
        super(CustomTrainingModel, self).__init__(name=name, **kwargs)
    

    @abstractmethod
    def _build_model(self, name):
        pass # Pure virtual functions => Must be overridden in the derived classes


    @abstractmethod
    def _compute_loss_and_metrics(self, batch, is_training=False):
        pass # Pure virtual functions => Must be overridden in the derived classes


    @abstractmethod
    def predict(self, batch_images):
        pass # Pure virtual functions => Must be overridden in the derived classes


    @tf.function
    def train_step(self, batch):
        with tf.GradientTape() as tape:
            loss, display_results = self._compute_loss_and_metrics(batch, is_training=True)

        # Apply an optimization step
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return display_results


    @tf.function
    def test_step(self, batch):
        _, display_results = self._compute_loss_and_metrics(batch)
        return display_results
    
    # def _update_metrics(self, batch):
    #     batch_images, batch_tokens = batch
    #     predictions = self.predict(batch_images) 
    #     self.compiled_metrics.update_state(batch_tokens, predictions)
    #     return {m.name: m.result() for m in self.metrics}


class DBNet(CustomTrainingModel):
    def __init__(self, k=50, name='DBNet', **kwargs):
        super(DBNet, self).__init__(name=name, **kwargs)
        self.model = self._build_model(k, name)

    
    def _build_model(self, k=50, name='DBNet'):
        image_input = Input(shape=(None, None, 3), name='image')
        backbone = ResNet50(inputs=image_input, include_top=False, freeze_bn=True)
        
        C2, C3, C4, C5 = backbone.outputs
        in2 = ConvBnRelu(256, kernel_size=1, name='in2')(C2)
        in3 = ConvBnRelu(256, kernel_size=1, name='in3')(C3)
        in4 = ConvBnRelu(256, kernel_size=1, name='in4')(C4)
        in5 = ConvBnRelu(256, kernel_size=1, name='in5')(C5)
        
        # The pyramid features are up-sampled to the same scale and cascaded to produce feature F
        P5 = ConvBnRelu(64, kernel_size=3, name='P5_conv')(in5)
        P5 = UpSampling2D(8, name='P5_up')(P5) # 1 / 32 * 8 = 1 / 4
        
        out4 = Add(name='out4')([in4, UpSampling2D(2, name='in5_up')(in5)])
        P4 = ConvBnRelu(64, kernel_size=3, name='P4_conv')(out4)
        P4 = UpSampling2D(4, name='P4_up')(P4) # 1 / 16 * 4 = 1 / 4
        
        out3 = Add(name='out3')([in3, UpSampling2D(2, name='out4_up')(out4)])
        P3 = ConvBnRelu(64, kernel_size=3, name='P3_conv')(out3)
        P3 = UpSampling2D(2, name='P3_up')(P3) # 1 / 8 * 2 = 1 / 4
        
        out2 = Add(name='out2')([in2, UpSampling2D(2, name='out3_up')(out3)])
        P2 = ConvBnRelu(64, kernel_size=3, name='P2_conv')(out2) # 1 / 4
        
        # Calculate DBNet maps
        fuse = Concatenate(name='fuse')([P2, P3, P4, P5]) # (batch_size, /4, /4, 256)
        binarize_map = DeconvolutionalMap(64, name='probability_map')(fuse)
        threshold_map = DeconvolutionalMap(64, name='threshold_map')(fuse)
        thresh_binary = Lambda( 
            lambda x: 1 / (1 + tf.exp(-k * (x[0] - x[1]))), # b_hat = 1 / (1 + e^(-k(P - T)))
            name = 'approximate_binary_map'
        )([binarize_map, threshold_map]) 
        return tf.keras.Model(inputs=image_input, outputs=[binarize_map, threshold_map, thresh_binary], name=name)


    @tf.function
    def _compute_loss_and_metrics(self, batch, is_training=False):
        y_pred = self.model(batch[0], training=True)
        loss = self.loss(batch[1], y_pred)
        # metrics = self._update_metrics(batch) # Update training display result
        return loss, {'loss': loss}


    def predict(self, batch_images):
        pass
