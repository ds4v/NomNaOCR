# https://www.tensorflow.org/text/tutorials/transformer
# https://keras.io/examples/nlp/neural_machine_translation_with_transformer
# https://keras.io/examples/vision/image_captioning
import sys
sys.path.append('..')

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from models import EncoderDecoderModel


class TransformerOCR(EncoderDecoderModel):
    def __init__(self, encoder, decoder, data_handler, name='TransformerOCR'):
        super(TransformerOCR, self).__init__(encoder, decoder, data_handler, name)
        self.final_layer = Dense(data_handler.char2num.vocab_size())

    def call(self, inputs, training=None):
        pass