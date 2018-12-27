import tensorflow as tf

def downsample(inputs, filters, rate, kernel_size, conv_stride, trainable):
    """down sampling block"""
    with tf.name_scope('downsample'):
        conv0 = tf.layers.conv2d(
            inputs=inputs, filters=filters, kernel_size=kernel_size,
            strides=conv_stride, padding='valid', activation=tf.nn.relu, trainable=trainable)
        conv1 = tf.layers.conv2d(
            inputs=conv0, filters=filters, kernel_size=kernel_size,
            strides=conv_stride, padding='valid', activation=tf.nn.relu, trainable=trainable)
        half = tf.layers.max_pooling2d(conv1, rate, rate)
    return conv1, half

def upsample(inputs, reference, filters, rate, kernel_size, conv_stride, trainable):
    """down sampling block"""
    with tf.name_scope('upsample'):
        reference_size = int(reference.get_shape()[1])

        tconv0 = tf.layers.conv2d_transpose(
            inputs=inputs, filters=filters, kernel_size=rate, strides=rate,
            padding='valid', activation=None, trainable=trainable)
        tconv0_size = int(tconv0.get_shape()[1])

        print('inputs:{}'.format(inputs.get_shape()))
        print('tconv:{}'.format(tconv0.get_shape()))
        print('reference:{}'.format(reference.get_shape()))

        # assuming reference_size > tconv0_size
        assert reference_size >= tconv0_size, '{} >= {}'.format(reference_size, tconv0_size)
        diff = reference_size - tconv0_size
        diff_half = tf.cast(diff/2, tf.int32)

        concatenated = tf.concat([tconv0, tf.image.crop_to_bounding_box(
            reference, diff_half, diff_half, tconv0_size, tconv0_size)], axis=-1)
        print('concatenated:{}'.format(concatenated.get_shape()))

        conv0 = tf.layers.conv2d(
            inputs=concatenated, filters=filters, kernel_size=kernel_size,
            strides=conv_stride, padding='valid', activation=tf.nn.relu, trainable=trainable)
        conv1 = tf.layers.conv2d(
            inputs=conv0, filters=filters, kernel_size=kernel_size,
            strides=conv_stride, padding='valid', activation=tf.nn.relu, trainable=trainable)
    return conv1

def encoder(inputs, filters_first, n_downsample, rate, kernel_size, conv_stride, trainable):
    """encoder block"""
    res_list = list()
    next_inputs = inputs
    next_filters = filters_first

    with tf.name_scope('encoder'):
        for i in range(n_downsample):
            print(next_inputs.get_shape())
            res, downsampled = downsample(
                next_inputs, next_filters, rate, kernel_size, conv_stride, trainable
            )
            res_list.append(res)

            next_inputs = downsampled
            next_filters = int(rate * next_filters)

    return res_list, downsampled

def decoder(inputs, res_list, rate, kernel_size, conv_stride, trainable):
    """decoder block"""
    filters_first = inputs.get_shape()[-1]
    next_inputs = inputs
    next_filters = filters_first
    print()

    with tf.name_scope('decoder'):
        for i in range(len(res_list)):
            print(next_inputs.get_shape())
            upsampled = upsample(
                next_inputs, res_list[-1], next_filters, rate, kernel_size, conv_stride, trainable
            )

            next_inputs = upsampled
            next_filters = int(int(next_filters)/2)
            del res_list[-1]

    return upsampled

def unet(inputs, filters_first, n_downsample, rate, kernel_size, conv_stride, trainable):
    '''
    this func represents unet
    '''
    with tf.name_scope('unet'):
        res_list, downsampled = encoder(
            inputs, filters_first, n_downsample, rate, kernel_size, conv_stride, trainable
        )
        output = decoder(
            downsampled, res_list, rate, kernel_size, conv_stride, trainable
        )

    # res_list must be empty because all of them are supposed to be consumed
    assert not res_list
    return output

def unet_based_annotator(input_, n_filter_first, n_downsample, rate, kernel_size, conv_stride, trainable):
    '''
    this function represents a part of model which will produce annotation or segmentation
    '''
    unet_out = unet(input_, n_filters_first, n_downsample, rate, kernel_size, conv_stride, trainable)
    seg = tf.layers.conv2d(inputs=unet_out, filters=1, kernel_size=1, activation=None, trainable=trainable)