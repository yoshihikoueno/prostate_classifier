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


def get_estimator(model_fn, model_dir=model_dir, save_interval=100, params=None):
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
            "unet_filters_first":params_list[0],
            "unet_n_downsample":params_list[1],
            "kernel_size":params_list[2],
            "conv_stride":params_list[3],
            "unet_rate":params_list[4],
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
        dimensions=[ (50, 80), (3, 7), (3, 4), (1, 2), ],
        x0=[64, 5, 3, 1,],
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
