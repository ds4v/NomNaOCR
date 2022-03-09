from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D, Reshape


def _reshape_for_cnn(last_cnn_layer, dim_to_keep):
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


def custom_cnn(config, image_input, dim_to_keep=-1):
    # Convolution layer with BatchNormalization and LeakyReLU activation
    def _conv_bn_leaky(input_layer, filters, block_name, conv_idx, is_last=False):
        x = Conv2D(
            filters = filters,
            kernel_size = (2, 2) if is_last else (3, 3),
            padding = 'valid' if is_last else 'same',
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
            x = _conv_bn_leaky(input_layer, filters, block_name, conv_idx + 1)
        x = MaxPooling2D(pool_size, name=f'{block_name}_pool')(x)

    # Last Convolution has 2x2 kernel with no padding and no followed MaxPooling layer
    x = _conv_bn_leaky(x, filters, 'final', '', True)
    return _reshape_for_cnn(x, dim_to_keep)


def imagenet_model(model_name, input_shape, get_layer, dim_to_keep=-1):
    base_model = eval('tf.keras.applications.' + model_name)
    base_model = base_model(input_shape=input_shape, weights=None, include_top=False)
    x = base_model.get_layer(name=get_layer if get_layer else base_model.layers[-1]).output
    return _reshape_for_cnn(x, dim_to_keep)
