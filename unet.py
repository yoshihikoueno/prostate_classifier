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
    "unet_n_downsample":3,
    "kernel_size":3,
    "conv_stride":1,
    "unet_rate":2,
}

def model_fn(features, labels, mode, params, config):
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
        reference_size = int(reference.get_shape()[1])

        tconv0 = tf.layers.conv2d_transpose(
            inputs=inputs, filters=filters, kernel_size=rate, strides=rate, padding='valid', activation=None)
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
            inputs=concatenated, filters=filters, kernel_size=kernel_size, strides=conv_stride, padding='valid', activation=tf.nn.relu)
        conv1 = tf.layers.conv2d(
            inputs=conv0, filters=filters, kernel_size=kernel_size, strides=conv_stride, padding='valid', activation=tf.nn.relu)
        return conv1

    def encoder(inputs, filters_first, n_downsample, rate, kernel_size, conv_stride):
        """encoder block"""
        res_list = list()
        next_inputs = inputs
        next_filters = filters_first

        for i in range(n_downsample):
            print(next_inputs.get_shape())
            res, downsampled = downsample(next_inputs, next_filters, rate, kernel_size, conv_stride)
            res_list.append(res)

            next_inputs = downsampled
            next_filters = int(rate * next_filters)

        return res_list, downsampled

    def decoder(inputs, res_list, rate, kernel_size, conv_stride):
        """decoder block"""
        filters_first = inputs.get_shape()[-1]
        next_inputs = inputs
        next_filters = filters_first
        print()

        for i in range(len(res_list)):
            print(next_inputs.get_shape())
            upsampled = upsample(next_inputs, res_list[-1], next_filters, rate, kernel_size, conv_stride)

            next_inputs = upsampled
            next_filters = int(int(next_filters)/2)
            del res_list[-1]

        return upsampled

    def unet(inputs, filters_first, n_downsample, rate, kernel_size, conv_stride):
        res_list, downsampled = encoder(inputs, filters_first, n_downsample, rate, kernel_size, conv_stride)
        output = decoder(downsampled, res_list, rate, kernel_size, conv_stride)

        # res_list must be empty because all of them are supposed to be consumed
        assert not res_list
        return output

    if not params.keys():
        params = default_params

    unet_out = unet(features, params["unet_filters_first"], params["unet_n_downsample"], params["unet_rate"], params["kernel_size"], params["conv_stride"])
    seg = tf.layers.conv2d(inputs=unet_out, filters=1, kernel_size=1, activation=None)

    predictions = {
        "prediction": tf.sigmoid(seg),
    }

    seg_size = tf.cast(seg.get_shape()[1], tf.int32)

    # crop labes and features
    # assuming they have the same sizes
    label_size = tf.cast(labels.get_shape()[1], tf.int32)
    diff_half = tf.cast(( label_size-seg_size )/2, tf.int32)
    labels = tf.image.crop_to_bounding_box(labels, diff_half, diff_half, seg_size, seg_size)
    features_cropped = tf.image.crop_to_bounding_box(features, diff_half, diff_half, seg_size, seg_size)

    labels_int = tf.cast(tf.greater_equal(labels, 0.5), tf.int64)
    loss = tf.losses.sigmoid_cross_entropy(labels_int, logits=seg)

    tf.summary.image("input", features_cropped)
    tf.summary.image("seg_raw", seg)
    tf.summary.image("seg_sigmoid", predictions['prediction'])
    tf.summary.image("seg_int", tf.cast(tf.greater_equal(predictions['prediction'], 0.5), tf.int64)*255 )
    tf.summary.image("label", labels)
    tf.summary.image("label_int", labels_int*255)

    summary_op = tf.summary.merge_all()
    summary_saver_hook = tf.train.SummarySaverHook(save_steps=config.save_summary_steps, output_dir=config.model_dir, summary_op=summary_op)
    # Estimators will save summaries while training session but not in eval or predict,
    #  so saver hook above is useful for eval and predict

    # Configure the Prediction Op (for PREDICT mode)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, prediction_hooks=[summary_saver_hook])

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, training_hooks=[])

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["prediction"]
        )
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops, evaluation_hooks=[summary_saver_hook]
    )
