"""
this module is a cancer classifier model
which contains
 1. model
 2. functions for train/eval

this classifier tries to classify cancers
into 5 classes.
"""

import tensorflow as tf
import tio
import skopt
import utility as util
import logging
logging.getLogger().setLevel(logging.INFO)

train_dir = "data/train"
eval_dir = "data/eval"
model_dir = "summary"


def model_fn(features, labels, mode, params):
    """
    this function is a model_fn for tensorflow
    """
    if not params.keys():
        params = {
            "n_conv": 1,
            "kernel_size": 3,
            "n_filter": 6,
            "n_dense": 1,
            "dense_units": 10,
            "learn_rate_init": 0.01,
            "learn_rate_rate": 0.98,
            "steps": 100,
        }

    input_layer = tf.reshape(features, [-1, tio.FLAGS.final_image_size, tio.FLAGS.final_image_size, 1])
    tf.summary.image("input", input_layer)

    # stack of conv layers
    input_to_conv = input_layer
    for i in range(params["n_conv"]):
        conv = tf.layers.conv2d(
            inputs=input_to_conv,
            filters=params["n_filter"],
            kernel_size=params["kernel_size"],
            activation=tf.nn.relu,
        )
        input_to_conv = conv
    conv = tf.layers.flatten(conv)

    # stack of dense layers
    input_to_dense = conv
    for i in range(params["n_dense"]):
        dense = tf.layers.dense(
            inputs=input_to_dense, units=params["dense_units"], activation=tf.nn.relu
        )
        input_to_dense = dense

    # output layer
    logits = tf.layers.dense(inputs=dense, units=5)
    loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)

    summary_op = tf.summary.merge_all()

    predictions = {
        "classes": tf.argmax(logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
    }

    # Configure the Prediction Op (for PREDICT mode)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        learning_rate = tf.train.exponential_decay(
            params["learn_rate_init"],
            tf.train.get_global_step(),
            params["steps"],
            params["learn_rate_rate"],
            True,
        )
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, training_hooks=[])

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"]
        )
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops
    )



def get_estimator(model_dir=model_dir, save_interval=100, params=None):
    """
    this function returns Estimator
    Args:
        model_dir: (str) directory where chechpoints and summaries will be saved
    """
    config_session = tf.ConfigProto(
        inter_op_parallelism_threads=tio.FLAGS.cores,
        intra_op_parallelism_threads=tio.FLAGS.cores,
        gpu_options=tf.GPUOptions(allow_growth=True)
    )
    config = tf.estimator.RunConfig(
        save_checkpoints_steps=save_interval,
        save_summary_steps=save_interval,
        session_config=config_session,
    )
    return tf.estimator.Estimator(
        model_fn=model_fn, model_dir=model_dir, config=config, params=params
    )


def train(steps=3000):
    """
    this function trains the model.
    Args:
        steps: the num of steps to train
    """
    estimator = get_estimator()
    eval_res, export_res = tf.estimator.train_and_evaluate(
        estimator=estimator,
        train_spec=tf.estimator.TrainSpec(
            input_fn=lambda: tio.input_func_train(train_dir), max_steps=steps
        ),
        eval_spec=tf.estimator.EvalSpec(
            input_fn=lambda: tio.input_func_test(eval_dir), throttle_secs=0
        ),
    )
    if eval_res is None:
        eval_res = estimator.evaluate(
            input_fn=lambda: tio.input_func_test(eval_dir)
        )
    print(eval_res)
    return eval_res


def hyperparameter_optimize(output="hyper_opt_res", max_steps=10000, n_calls=1000):
    """
    this function will perform hyperperameter optimization
    to the model and save the result to "output" file

    Args:
        output: the output directory where to save the results
        max_steps: the max steps for each trial
        n_calls: the max num for trials

    This function will perform training for (max_steps * n_call) steps totally.
    """
    output = output + "/" if output[-1] != "/" else output

    def wrapper(params_list):
        """wrapper func for get_estimator"""
        nonlocal counter
        params = {
            "n_conv": params_list[0],
            "kernel_size": int(params_list[1]),
            "n_filter": params_list[2],
            "n_dense": params_list[3],
            "dense_units": params_list[4],
            "learn_rate_init": params_list[5],
            "learn_rate_rate": params_list[6],
            "steps": params_list[7],
        }

        print("Trial {} / {}".format(counter, n_calls))
        print(params)
        print()

        estimator = get_estimator(
            model_dir=output + util.config_to_file_name(params),
            save_interval=500,
            params=params,
        )
        early_stop = tf.contrib.estimator.stop_if_no_decrease_hook(
            estimator=estimator,
            metric_name="loss",
            max_steps_without_decrease=estimator.config.save_checkpoints_steps * 8,
            run_every_secs=None,
            run_every_steps=estimator.config.save_checkpoints_steps,
        )

        try:
            eval_res, export_res = tf.estimator.train_and_evaluate(
                estimator=estimator,
                train_spec=tf.estimator.TrainSpec(
                    input_fn=lambda: tio.input_func_train(train_dir),
                    max_steps=max_steps,
                    hooks=[early_stop],
                ),
                eval_spec=tf.estimator.EvalSpec(
                    input_fn=lambda: tio.input_func_test(eval_dir),
                    throttle_secs=0,
                ),
            )
            if eval_res is None:
                eval_res = estimator.evaluate(
                    input_fn=lambda: tio.input_func_test(eval_dir)
                )
        except tf.train.NanLossDuringTrainingError:
            print("Diverged")
            eval_res = {"accuracy": -1}

        counter += 1

        with open(output + util.config_to_file_name(params) + "/result", mode="w") as f:
            f.write(str(eval_res["accuracy"]))

        return -eval_res["accuracy"]

    counter = 0
    res = skopt.gp_minimize(
        func=wrapper,
        dimensions=[
            (1, 4),
            (3, 6),
            (6, 15),
            (1, 6),
            (10, 30),
            (0.01, 0.1),
            (0.97, 0.99),
            (100, 200),
        ],
        x0=[1, 6, 10, 1, 30, 0.05, 0.99, 100],
        n_calls=n_calls,
    )

    with open(output + "result", mode="w") as f:
        f.write(str(res))

    return res


def predict(image):
    """
    this function classify a cancer
    """
    classifier = get_estimator()
    result = classifier.predict(input_fn=lambda: tio.input_func_predict(image))
    return result
