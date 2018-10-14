"""
this module is a cancer classifier model
which contains
 1. model

"""

import tensorflow as tf
import tio
import skopt
import utility as util
import logging
logging.getLogger().setLevel(logging.INFO)

default_params = {
    "unet_filters_first":64,
    "unet_n_downsample":5,
    "kernel_size":3,
    "conv_stride":1,
    "unet_rate":2,
}

def model_fn(features, labels, mode, params):
    """
    this function is a model_fn for tensorflow
    """
    def downsample(inputs, filters, rate, kernel_size, conv_stride):
        """down sampling block"""
        conv0 = tf.layers.conv2d(
            inputs=inputs, filters=filters, kernel_size=kernel_size, strides=conv_stride, padding='valid', activation=tf.nn.relu)
        conv1 = tf.layers.conv2d(
            inputs=conv0, filters=filters, kernel_size=kernel_size, strides=conv_stride, padding='valid', activation=tf.nn.relu)
        half = tf.layers.max_pooling2d(conv1, rate, rate)
        return conv1, half

    def upsample(inputs, reference, filters, rate, kernel_size, conv_stride):
        """down sampling block"""
        reference_size = reference.get_shape()[0]

        tconv0 = tf.layers.conv2d_transpose(
            inputs=inputs, filters=filters, kernel_size=rate, strides=rate, padding='valid', activation=None)
        tconv0_size = tconv0.get_shape()[0]

        # assuming reference_size > tconv0_size
        assert reference_size > tconv0_size

        diff = reference_size - tconv0_size

        concatenated = tf.concat([tconv0, tf.image.crop_to_bounding_box(
            reference, diff/2, diff/2, tconv0_size, tconv0_size)], axis=-1)

        conv0 = tf.layers.conv2d(
            inputs=concatenated, filters=filters, kernel_size=kernel_size, strides=conv_stride, padding='valid', activation=tf.nn.relu)
        conv1 = tf.layers.conv2d(
            inputs=conv0, filters=filters, kernel_size=kernel_size, strides=conv_stride, padding='valid', activation=tf.nn.relu)
        return conv1

    def encoder(inputs, filters_first4, n_downsample, rate, kernel_size, conv_stride):
        """encoder block"""
        res_list = list()
        next_inputs = inputs
        next_filters = filters_first

        for i in range(n_downsample):
            res, downsampled = downsample(next_inputs, next_filters, kernel_size, conv_stride)
            res_list.append(res_list)

            next_inputs = downsampled
            next_filters = int(rate * next_filters)

        return res_list, downsampled

    def decoder(inputs, res_list, rate, kernel_size, conv_stride):
        """decoder block"""
        filters_first = inputs.get_shape()[-1]
        next_inputs = inputs
        next_filters = filters_first

        for i in range(len(res_list)):
            upsampled = upsample(next_inputs, res_list[-1], next_filters, kernel_size, conv_stride)

            next_inputs = upsampled
            next_filters = int(rate * next_filters)
            del res_list[-1]

    def unet(inputs, filters_first, n_downsample, rate, kernel_size, conv_stride):
        res_list, downsampled = encoder(inputs, filters_first, n_downsample, rate, kernel_size, conv_stride)
        output = decoder(downsampled, res_list, rate, kernel_size, conv_stride)

        # res_list must be empty because all of them are supposed to be consumed
        assert not res_list
        return output

    if not params.keys():
        params = default_params

    unet_out = unet(features, params["unet_filters_first"], params["unet_filters_first"], params["unet_rate"], params["kernel_size"], params["conv_stride"])
    seg = tf.layers.conv2d(inputs=unet_out, filters=2, kernel_size=1, activation=None)

    predictions = {
        "prediction": tf.argmax(seg, axis=-1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
    }

    tf.summary.image("input", features, )
    tf.summary.image("seg_1", seg[:,:,:,1])
    tf.summary.image("seg_2", seg[:,:,:,2])
    tf.summary.image("predict", predictions["prediction"])

    # Configure the Prediction Op (for PREDICT mode)
    if mode == tf.estimator.ModeKeys.PREDICT:
        tf.summary.merge_all()
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        tf.summary.merge_all()
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, training_hooks=[])

    # Add evaluation metrics (for EVAL mode)
    tf.summary.image("ground_truth", labels)
    tf.summary.merge_all()
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["prediction"]
        )
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops
    )
