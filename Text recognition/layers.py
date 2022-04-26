import tensorflow as tf
from tensorflow.keras.layers import (
    Convolution2D, MaxPooling2D, BatchNormalization, Dense, Multiply,
    Activation, LeakyReLU, Reshape, Permute, Lambda, RepeatVector
)


def get_imagenet_model(model_name, input_shape):
    # Pick a model from https://keras.io/api/applications
    base_model = eval('tf.keras.applications.' + model_name)
    return base_model(input_shape=input_shape, weights=None, include_top=False)


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
