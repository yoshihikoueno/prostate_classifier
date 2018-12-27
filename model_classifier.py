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
from components import unet_based_annotator as annotator
from components import encoder as cnn
logging.getLogger().setLevel(logging.INFO)

default_params = {
    "unet_filters_first":64,
    "unet_n_downsample":3,
    "kernel_size":3,
    "conv_stride":1,
    "unet_rate":2,
    "cnn_filters_first":16,
    "cnn_n_downsample":4,
    "cnn_rate":2,
    'cnn_kernel': 5,
}

def model_fn(features, labels, mode, params, config):
    """
    this function is a model_fn for tensorflow
    """

    if not params.keys():
        params = default_params

    seg = annotator(features['raw'], params["unet_filters_first"], params["unet_n_downsample"],
                    params["unet_rate"], params["kernel_size"], params["conv_stride"], False)

    _, cnn_out = cnn(seg, params['cnn_filters_first'], params['cnn_n_downsample'],
                     params['cnn_rate'], params['kernel_size'], params['conv_stride'], True)

    flat_cnn_out = tf.layers.flatten(cnn_out)
    logits = tf.layers.dense(flat_cnn_out, 5, tf.nn.relu)

    predictions = {
        "annotation": tf.sigmoid(seg),
        'group': tf.argmax(logits, axis=1),
    }

    # Configure the Prediction Op (for PREDICT mode)
    if mode == tf.estimator.ModeKeys.PREDICT:
        # Estimators will save summaries while training session but not in eval or predict,
        #  so saver hook above is useful for eval and predict
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    seg_size = tf.cast(seg.get_shape()[1], tf.int32)

    # crop labes and features
    # assuming they have the same sizes
    label_size = tf.cast(labels['annotation'].get_shape()[1], tf.int32)
    diff_half = tf.cast(( label_size-seg_size )/2, tf.int32)
    labels['annotation'] = tf.image.crop_to_bounding_box(labels['annotation'], diff_half, diff_half, seg_size, seg_size)
    features_cropped = tf.image.crop_to_bounding_box(features['raw'], diff_half, diff_half, seg_size, seg_size)

    labels_int = util.tf_threshold(labels['annotation'], 0.5, 1, tf.uint8)
    loss_seg = tf.losses.sigmoid_cross_entropy(labels_int, logits=seg)
    loss_group = tf.losses.sparse_softmax_cross_entropy(labels['group'], logits)
    loss = loss_group

    tf.summary.image("input", features_cropped)
    tf.summary.image("seg_sigmoid", predictions['annotation'])
    tf.summary.image("label", labels['annotation'])
    tf.summary.histogram('group_logits', logits)
    tf.summary.scalar('loss_seg', loss_seg)
    tf.summary.scalar('loss_group', loss_group)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, training_hooks=[])

    # Add evaluation metrics (for EVAL mode)
    summary_saver_hook = tf.train.SummarySaverHook(
        save_steps=config.save_summary_steps,
        output_dir=config.model_dir+'eval' if config.model_dir[-1]=='/' else config.model_dir+'/eval',
        summary_op=tf.summary.merge_all())
    # Estimators will save summaries while training session but not in eval or predict,
    #  so saver hook above is useful for eval and predict
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels['group'], predictions=predictions["group"]
        )
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops, evaluation_hooks=[summary_saver_hook]
    )
