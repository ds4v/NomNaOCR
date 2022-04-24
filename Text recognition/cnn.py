import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, LeakyReLU, MaxPool2D, 
    Reshape, Permute, Dense, Lambda, RepeatVector, Multiply
)


def get_imagenet_model(model_name, input_shape):
    # Pick a model from https://keras.io/api/applications
    base_model = eval('tf.keras.applications.' + model_name)
    return base_model(input_shape=input_shape, weights=None, include_top=False)


def custom_cnn(config, image_input, use_extra_conv=True):
    # Convolution layer with BatchNormalization and LeakyReLU activation
    def _conv_bn_leaky(input_layer, filters, block_name, conv_idx, use_pooling=False):
        x = Conv2D(
            filters = filters,
            kernel_size = (2, 2) if not use_pooling else (3, 3),
            padding = 'valid' if not use_pooling else 'same',
            kernel_initializer = 'he_uniform',
            name = f'{block_name}_conv{conv_idx}'
        )(input_layer)
        x = BatchNormalization(name=f'{block_name}_bn{conv_idx}')(x)
        return LeakyReLU(alpha=0.2, name=f'{block_name}_activation{conv_idx}')(x)

    # Generate Convolutional blocks by config
    for idx, (block_name, block_config) in enumerate(config.items()):
        num_conv, filters, pool_size = block_config.values()
        for conv_idx in range(num_conv):
            input_layer = image_input if idx + conv_idx == 0 else x
            x = _conv_bn_leaky(input_layer, filters, block_name, conv_idx + 1, pool_size)
        if pool_size is not None: x = MaxPool2D(pool_size, name=f'{block_name}_pool')(x)
    return x


def reshape_for_rnn(last_cnn_layer, dim_to_keep=-1):
    # Reshape accordingly before passing the output to the RNN part
    _, height, width, channel = last_cnn_layer.get_shape()
    if dim_to_keep == 1:
        target_shape = (height, width * channel)
    elif dim_to_keep == 2:
        target_shape = (width, height * channel)
    elif dim_to_keep == 3 or dim_to_keep == -1:
        target_shape = (height * width, channel)
    else:
        raise ValueError('Invalid dim_to_keep value')
    return Reshape(target_shape=(target_shape), name='rnn_input')(last_cnn_layer)


# https://pbcquoc.github.io/vietnamese-ocr (Vietnamese blog)
def visual_attention(feature_maps):
    _, timestep, input_dim = feature_maps.shape
    a = Permute((2, 1), name='dim_switching1')(feature_maps)
    a = Dense(timestep, activation='softmax', name='attention_scores')(a)
    a = Lambda(lambda x: tf.reduce_sum(x, axis=1), name='dim_reduction')(a)
    a = RepeatVector(input_dim, name='redistribute')(a)
    a = Permute((2, 1), name='dim_switching2')(a) 
    return Multiply(name='context_vector')([feature_maps, a])
