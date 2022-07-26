import tensorflow as tf
from tensorflow.keras.layers import (
    Convolution2D, MaxPooling2D, BatchNormalization, Dense, Multiply,
    Activation, LeakyReLU, Reshape, Permute, Lambda, RepeatVector
)


def custom_cnn(config, image_input, alpha=0):
    # Generate Convolutional blocks by config
    for idx, (block_name, block_config) in enumerate(config.items()):
        num_conv, filters, pool_size = block_config.values()
        for conv_idx in range(1, num_conv + 1):
            x = Convolution2D(
                filters = filters,
                kernel_size = (3, 3) if pool_size else (2, 2),
                padding = 'same' if pool_size else 'valid',
                kernel_initializer = 'he_uniform',
                name = f'{block_name}_conv{conv_idx}'
            )(image_input if idx + conv_idx == 1 else x)

            x = BatchNormalization(name=f'{block_name}_bn{conv_idx}')(x)
            if alpha > 0: x = LeakyReLU(alpha, name=f'{block_name}_act{conv_idx}')(x)
            else: x = Activation('relu', name=f'{block_name}_relu{conv_idx}')(x)

        if pool_size is not None: 
            x = MaxPooling2D(pool_size, name=f'{block_name}_pool')(x)
    return x


def reshape_features(last_cnn_layer, dim_to_keep=-1, name='cnn_features'):
    # Reshape accordingly before passing the output to the RNN or Transformer
    _, height, width, channel = last_cnn_layer.get_shape()
    if dim_to_keep == 1:
        target_shape = (height, width * channel)
    elif dim_to_keep == 2:
        target_shape = (width, height * channel)
    elif dim_to_keep == 3 or dim_to_keep == -1:
        target_shape = (height * width, channel)
    else:
        raise ValueError('Invalid dim_to_keep value')
    return Reshape(target_shape=(target_shape), name=name)(last_cnn_layer)


# https://pbcquoc.github.io/vietnamese-ocr (Vietnamese blog)
def visual_attention(feature_maps):
    _, timestep, input_dim = feature_maps.shape
    a = Permute((2, 1), name='dim_switching1')(feature_maps)
    a = Dense(timestep, activation='softmax', name='attention_scores')(a)
    a = Lambda(lambda x: tf.reduce_sum(x, axis=1), name='dim_reduction')(a)
    a = RepeatVector(input_dim, name='redistribute')(a)
    a = Permute((2, 1), name='dim_switching2')(a) 
    return Multiply(name='context_vector')([feature_maps, a])


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units, name='BahdanauAttention', **kwargs):
        super(BahdanauAttention, self).__init__(name=name, **kwargs)
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, enc_output, hidden):
        # encoder output shape == (batch_size, receptive_size, channels)
        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # attention_hidden_layer shape == (batch_size, receptive_size, units)
        attention_hidden_layer = self.W1(enc_output) + self.W2(hidden_with_time_axis)
        attention_hidden_layer = tf.nn.tanh(attention_hidden_layer)

        # score shape == (batch_size, receptive_size, 1)
        # This gives you an unnormalized score for each image feature.
        score = self.V(attention_hidden_layer)

        # attention_weights shape == (batch_size, receptive_size, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # context_vector shape after sum == (batch_size, channels)
        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights


class AdditiveAttention(tf.keras.layers.Layer):
    def __init__(self, units, name='AdditiveAttention', **kwargs):
        super(AdditiveAttention, self).__init__(name=name, **kwargs)
        self.W1 = Dense(units, use_bias=False)
        self.W2 = Dense(units, use_bias=False)
        self.attention = tf.keras.layers.AdditiveAttention()

    def call(self, query, value):
        w1_query, w2_key = self.W1(query), self.W2(value)
        context_vector, attention_weights = self.attention(
            inputs = [w1_query, value, w2_key],
            return_attention_scores = True,
        )
        return context_vector, attention_weights