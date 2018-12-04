"""
This module is a utility module named "tio" which
 stands for "Tensorflow Input / Output" and
 designed to support data preprocessing for tensorflow.
"""

import sys
import os
import multiprocessing
import regex
import tensorflow as tf
import utility as util

def _print_error(message, func_name):
    '''
    this function will print out error
    message
    '''
    print('{} in {}'.format(message, func_name))
    return

def input_func_train(data_dir, mode):
    """
    Args:
        data_dir: directory that stores data
        mode: either "classification", "annotation", or "both"
    Returns:
        tf.Dataset
    """
    mode_error_msg = "Unexpected mode specified"

    if mode == "classification":
        dataset = create_simple_ds_for_classification(data_dir)
    elif mode == "annotation":
        dataset = create_simple_ds_for_annotation(data_dir)
    elif mode == "both":
        dataset = create_simple_ds_for_both(data_dir)
    else:
        _print_error(mode_error_msg, sys._getframe().f_code.co_name)

    if mode == "classification":
        dataset = determine_channel_size(dataset, 1)
    elif mode == "annotation":
        dataset = determine_channel_size(dataset, 2)
    else:
        _print_error(mode_error_msg, sys._getframe().f_code.co_name)

    dataset = augment_ds(dataset)
    dataset = normalize(dataset)

    if mode == "annotation":
        dataset = divide_channels(dataset)

    dataset = dataset.shuffle(FLAGS.shuffle_buffer_size)
    dataset = dataset.batch(FLAGS.batch_size)
    dataset = dataset.prefetch(FLAGS.prefetch)
    dataset = dataset.repeat(None)
    return dataset


def determine_channel_size(dataset, channel_size):
    '''
    this function will determines
    the channel size in the given dataset

    note that default channel_size is 2 because
    feature image and label image are concatenated
    '''
    dataset = dataset.map(
        lambda image, *extra: (
            util.tf_determine_image_channel(image, channel_size), *extra
        ) if extra else util.tf_determine_image_channel(image, channel_size),
        FLAGS.cores,)
    return dataset


def divide_channels(dataset):
    """
    this function divides images which has 2 channel,
    which are supposed to be features channel and
    labels channel, into separate data with
    tuple object.

    Args:
        dataset: dataset consists of images with 2 channel
    Returns:
        dataset: tuple(features, labels), both are 1 channel
            images
    """
    dataset = dataset.map(
        lambda raw_annotation: (
            raw_annotation[:, :, 0:1], raw_annotation[:, :, 1:]),
        FLAGS.cores,)
    return dataset


def normalize(dataset):
    """
    this function map image
    [0,255] -> [0,1]
    """
    dataset = dataset.map(
        lambda image, *extra: (tf.div(tf.cast(image, tf.float32), 255.0),
                               *extra) if extra else tf.div(tf.cast(image, tf.float32), 255.0),
        FLAGS.cores,
    )
    return dataset


def create_simple_ds_for_annotation(data_dir, label_value=(255, 0, 0)):
    """
    this function creates simple ds for segmentation

    Returns:
        dataset = (image, patient_id, img_id)
    """
    data_dir = data_dir if data_dir[-1] == "/" else data_dir + "/"

    dataset = tf.data.Dataset.list_files(
        file_pattern=data_dir + "annotation/*/*/*", shuffle=True)
    # we need to add '*/*/*' at the end of data_dir
    #  because we suppose directory structure like below.
    #   data/train/annotation/cancer_annotations/5468464/23.jpg
    #   data/train/annotation/healthy_cases/5468464/23.jpg
    # NOTE: data_dir = "data/train/"

    # prepare the paths
    dataset = dataset.map(
        lambda path_annotation: (
            util.tf_get_raw(path_annotation),
            tf_get_patient_img_from_path(path_annotation)),
        num_parallel_calls=FLAGS.cores,)
    # now dataset = (path_raw, path_anno, patient_id, image_id)

    # existence check
    dataset = dataset.filter(lambda raw, annotated, *others: util.tf_exists([raw, annotated]))

    # load and decode
    dataset = dataset.map(
        lambda raw, annotated, *others: (
            tf.image.decode_image(tf.read_file(raw)),
            tf.image.decode_image(tf.read_file(annotated)),
            *others,
        ),
        num_parallel_calls=FLAGS.cores,)

    # change the colorscale and extract labels
    dataset = dataset.map(
        lambda raw, annotated, *others: (
            tf.image.rgb_to_grayscale(raw),
            util.tf_extract_label(annotated, label_value),
            *others
        ),
        num_parallel_calls=FLAGS.cores,)

    # determine the size of the image
    dataset = dataset.map(
        lambda raw, annotation, *others: (
            util.tf_determine_image_size(raw),
            util.tf_determine_image_size(annotation),
            *others,
        ),
        num_parallel_calls=FLAGS.cores,)

    # crop
    dataset = dataset.map(
        lambda raw, annotation, *others: (
            tf.image.resize_images(raw, [FLAGS.init_image_size, FLAGS.init_image_size]),
            tf.image.resize_images(annotation, [FLAGS.init_image_size, FLAGS.init_image_size]),
            *others,
        ),
        num_parallel_calls=FLAGS.cores,)

    # combine
    dataset = dataset.map(
        lambda raw, annotation, *others: (
            tf.concat([raw, annotation], 2),
            *others,
        ))
    # now dataset = (img_raw_annotation, patient_id, image_id)

    # dictionalize
    dataset = dataset.map(
        lambda combined_img, patient_id, img_id: {
            'image': combined_img,
            'patient_id': patient_id,
            'img_id': img_id,
        })
    return dataset

def query_group(patient_id, data_dir='./data'):
    '''
    this func tries to figure out the cancer class
    for given patient id
    '''
    group_dir = '{}/RAW/cancer_cases_grouped'.format(data_dir)
    if isinstance(patient_id, int):
        patient_id = str(patient_id)

    for group in os.listdir(group_dir):
        if patient_id in os.listdir('{}/{}'.format(group_dir, group)):
            group_int = int(regex.sub(r'.*(\d+).*', r'\1', group))
            return group_int
    print('Error: failed to retrieve group for patient_id: {}'.format(patient_id))
    return -1

def create_simple_ds_for_both(data_dir, label_value=(255, 0, 0)):
    '''
    this function creates dataset that contatins
    (Raw MRI, Cancer Annotation, Cancer Stage Class)
    '''
    dataset = create_simple_ds_for_annotation(data_dir, label_value)
    dataset = dataset.map(
        lambda dictionary:
    )
    pass

def lambda_for_dict(src_dst_lambda, target_dir):
    '''
    this func provides lambda functionatily
    which is especially useful when you
    want to apply some operation on
    the list of dictionaries, where
    you maybe want to apply some operation
    on just a part of the dictionary and
    want others to be the same.
    Args:
        arg_lambda: list of arg and lambda
            arg must be an element of keys
            arg_lambda = [(key0, lambda0), ...]
    '''
    if isinstance(arg_lambdas[0], str):
        src, dst, operation = arg_lambdas
        target_dir[arg] = operation(target_dir[arg])
        return target_dir
    else:
        for arg, operation in arg_lambdas:
            target_dir[arg] = operation(target_dir[arg])
        return target_dir

def create_simple_ds_for_classification(data_dir):
    """
    this function creates simple ds for segmentation
    note that "create_simple_ds_for_classification"
    and "create_simple_ds_for_annotation" cannot be
    merged together due to the difference in directory
    structures.

    Returns:
        Dataset consits of dicts
        each dict has key[image, group, patient_id, img_id]
    """
    data_dir = data_dir if data_dir[-1] == "/" else data_dir + "/"

    dataset = tf.data.Dataset.list_files(
        file_pattern=data_dir + "classification/*/*/*", shuffle=True)
    # we need to add '*/*/*' at the end of data_dir
    #  because we suppose directory structure like below.
    #   data/train/Group1/5468464/23.jpg
    # NOTE: data_dir = "data/train/"

    dataset = dataset.map(tf_get_group_patient_img_from_path, FLAGS.cores)
    dataset = dataset.map(
        lambda image, group, patient_id, img_id: {
            'image': tf.image.decode_image(tf.read_file(image)),
            'group': group,  # Group Number
            'patient_id': patient_id,
            'img_id': img_id,
        },
        FLAGS.cores,
    )
    dataset = dataset.map(
        lambda dictionary: lambda_for_dict((
            'image',
            lambda image: tf.resize_images(
                tf.image.rgb_to_grayscale(util.tf_determine_image_size(image)),
                [FLAGS.init_image_size] * 2,
            ))),
        FLAGS.cores,
    )
    return dataset


def tf_get_group_patient_img_from_path(path):
    """wrapper func"""
    return tf.py_func(
        get_group_patient_img_from_path, [path], [
            tf.string, tf.int64, tf.int64, tf.int64]
    )


def tf_get_patient_img_from_path(path):
    """wrapper func"""
    return tf.py_func(
        get_group_patient_img_from_path, [
            path], [tf.string, tf.int64, tf.int64]
    )


def get_group_patient_img_from_path(path):
    """
    this function gets label for the image from path(str)

    images are supposed to have a path lile below
     'something/<group no.>/<patient ID>/index.<extension>'
    we can get the label from <group no.>

    Args:
        str path

    Return:
        (path, group, patient_id, img_id)
    """

    # TF converts strings into bytes object internally,
    #  so we have to convert them back to string.
    if isinstance(path, bytes):
        path = path.decode()

    group = regex.sub(
        r"^.*/Group(\d)/(\d+)/(\d+)\.(jpg|jpeg|png|bmp)", r"\1", path)

    group = int(group)
    _, patient_id, img_id = get_patient_img_from_path(path)

    # group number starts from 1
    group -= 1
    return (path, group, patient_id, img_id)


def get_patient_img_from_path(path):
    """
    this function gets label for the image from path(str)

    images are supposed to have a path lile below
     'something/<patient ID>/index.<extension>'

    Args:
        str path

    Return:
        (path, patient_id, img_id)
    """

    # TF converts strings into bytes object internally,
    #  so we have to convert them back to string.
    if isinstance(path, bytes):
        path = path.decode()

    patient_id = regex.sub(
        r"^.*/(\d+)/(\d+)\.(jpg|jpeg|png|bmp)", r"\1", path
    )
    img_id = regex.sub(
        r"^.*/Group(\d)/(\d+)/(\d+)\.(jpg|jpeg|png|bmp)", r"\3", path)

    patient_id, img_id = list(map(int, [patient_id, img_id]))

    return (path, patient_id, img_id)


def augment_ds(ds):
    """
    this function performs data augmentation
    to given dataset and returns dataset
    """

    def augment_static(image):
        """
        performs augmentation to given image
        """
        image = tf.image.random_contrast(image, 0.5, 1.5)
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, 0.5)
        return image

    def augment_dynamic(image):
        """
        performs augmentation by cropping/resizing
        given image
        """
        image = tf.image.resize_images(
            tf.image.crop_to_bounding_box(
                image,
                offset_height=tf.div(
                    FLAGS.init_image_size
                    - FLAGS.intermediate_image_size
                    + tf.random_uniform([], -10, 10, dtype=tf.int64),
                    2,
                ),
                offset_width=tf.div(
                    FLAGS.init_image_size
                    - FLAGS.intermediate_image_size
                    + tf.random_uniform([], -10, 10, dtype=tf.int64),
                    2,
                ),
                target_height=FLAGS.intermediate_image_size
                + tf.random_uniform([], -10, 10, dtype=tf.int64),
                target_width=FLAGS.intermediate_image_size
                + tf.random_uniform([], -10, 10, dtype=tf.int64),
            ),
            [FLAGS.final_image_size] * 2,
        )
        return image

    def augment_warp(image, n_points=10, width_index=0, height_index=1, threshold=5, default=0.0):
        '''
        this function will perfom data augmentation
        using Non-affine transformation, namely
        image warping.
        Currently, only square images are supported

        Args:
            image: input image
            n_points: the num of points to take for image warping
            width_index: index of width, set this to 1 if batched 0 otherwise normally
            height_index: index of height
            threshold: threshold to judge random value inappropriate
                diff values with its abs heigher width/threshold will be judged inappropriate
            default: value to be used when random value is over threshold
        Return:
            warped image
        '''
        width = int(image.get_shape()[width_index])
        height = int(image.get_shape()[height_index])
        assert width == height

        default = tf.cast(default, tf.float32)

        raw = tf.random_uniform([1, n_points, 2], 0.0, tf.cast(width, tf.float32), dtype=tf.float32)
        diff = tf.random_normal([1, n_points, 2], mean=0.0, dtype=tf.float32)
        # ensure that diff is not too big
        diff = default*tf.cast(tf.greater(tf.abs(diff), width/threshold), tf.float32) + \
            diff*tf.cast(tf.less_equal(tf.abs(diff), width/threshold), tf.float32)

        # expand dimension to meet the requirement of sparse_image_warp
        image = tf.expand_dims(image, 0)

        image = tf.contrib.image.sparse_image_warp(
            image=image,
            source_control_point_locations=raw,
            dest_control_point_locations=raw+diff,
        )[0]
        # sparse_image_warp function will return a tuple
        # (warped image, flow_field)

        # shrink dimension
        image = image[0,:,:,:]
        return image

    def fused_augmentation(image, label, augment_func):
        '''
        this function is a fuse function that allows
        to pass both data and label
        Returned values are supposed to be sliced

        Args:
            image: image data
            label: label data
            augment_func: function(s) used to augment
        Return:
            Tuple(list of images, list of labels)
        '''
        if isinstance(augment_func, list) or isinstance(augment_func, tuple):
            for func in augment_func:
                image = func(image)
        else:
            images = augment_func(image)

        # for some cases, like annotation, we don't need
        #  label data
        if label is None:
            return image

        return (image, label)

    ds = ds.map(
        lambda image, *
        extra: fused_augmentation(
            image, extra[0], [augment_dynamic, augment_static])
        if extra else fused_augmentation(image, None, [augment_dynamic, augment_static, augment_warp]),
        num_parallel_calls=FLAGS.cores,
    )
    return ds


def resize_image(ds):
    """
    make images smaller so that they can be handled by DNN
    """
    ds = ds.map(
        lambda image, *extra: (
            tf.image.resize_images(
                tf.image.crop_to_bounding_box(
                    image,
                    offset_height=tf.div(
                        FLAGS.init_image_size - FLAGS.intermediate_image_size, 2),
                    offset_width=tf.div(
                        FLAGS.init_image_size - FLAGS.intermediate_image_size, 2),
                    target_height=FLAGS.intermediate_image_size,
                    target_width=FLAGS.intermediate_image_size,
                ),
                [FLAGS.final_image_size] * 2,
            ),
            group,
        ) if extra else tf.image.resize_images(
            tf.image.crop_to_bounding_box(
                image,
                offset_height=tf.div(
                    FLAGS.init_image_size - FLAGS.intermediate_image_size, 2),
                offset_width=tf.div(
                    FLAGS.init_image_size - FLAGS.intermediate_image_size, 2),
                target_height=FLAGS.intermediate_image_size,
                target_width=FLAGS.intermediate_image_size,),
            [FLAGS.final_image_size] * 2,),
        FLAGS.cores,
    )
    return ds


def dataset_from_tfrecord(file_name):
    """
    this function generate dataset from tfrecord
    Args:
        file_name: file names for tfrecord
    Returns: dataset (unbatched)
    """
    file_name = [file_name] if not isinstance(file_name, list) else file_name
    file_num = len(file_name)

    dataset = tf.data.Dataset.from_tensor_slices(file_name).interleave(
        lambda x: tf.data.TFRecordDataset(x, buffer_size=1024 * 1024).map(
            lambda example: tf.parse_single_example(
                example,
                features={
                    "word": tf.FixedLenFeature([FLAGS.max_words], tf.string),
                    "label": tf.FixedLenFeature([FLAGS.max_words], tf.int64),
                },
            ),
            num_parallel_calls=FLAGS.cores,
        ),
        cycle_length=file_num,
        block_length=1,
    )
    dataset = dataset.map(
        lambda feature_dict: (
            {"word": feature_dict["word"]}, feature_dict["label"]),
        num_parallel_calls=FLAGS.cores,
    )
    return dataset


def is_tfrecord(doc_name):
    """this funciton judges if a file is a tfrecord"""
    if isinstance(doc_name, str):
        return doc_name.find(".tfrecords") != -1
    if isinstance(doc_name, list) and doc_name:
        return is_tfrecord(doc_name[0])
    return False


def input_func_test(data_dir, mode):
    """
    this function generates dataset for tensorflow Estimators
     leveraging dataset_from_doc function which is supposed to be
     base function for generating dataset.
    Args:
        doc_name: a file name for a document or a tfrecord file (must end with ".tfrecords")
        mode: mode 'annotation', 'classification', or 'both'
            annotation: raw MRI and annotation
            classification: raw MIR and label
            both: raw MRI and annotation and label
    Returns:
        tf.Dataset
            touple(dict, label)
    """
    if mode == "classification":
        dataset = create_simple_ds_for_classification(data_dir)
    elif mode == "annotation":
        dataset = create_simple_ds_for_annotation(data_dir)
    elif mode == 'both':
        dataset = create_simple_ds_for_both(data_dir)
    else:
        print("Unexpected mode specified in tio.input_func_train: {}".format(mode))

    dataset = resize_image(dataset)
    dataset = normalize(dataset)
    if mode == "classification":
        dataset = determine_channel_size(dataset, 1)
    elif mode == "annotation":
        dataset = determine_channel_size(dataset, 2)
    else:
        _print_error(mode_error_msg, sys._getframe().f_code.co_name)
    if mode == "annotation":
        dataset = divide_channels(dataset)

    dataset = dataset.batch(FLAGS.batch_size)
    dataset = dataset.prefetch(FLAGS.prefetch)
    return dataset


def input_func_predict(image_path):
    """
    this function converts string into dataset
    so that it can be passed to the model.
    """
    dataset = tf.data.Dataset.from_tensors(image_path)
    dataset = dataset.map(tf.read_file)
    dataset = dataset.map(tf.image.decode_image)
    dataset = resize_image(dataset)
    dataset = normalize(dataset)
    dataset = determine_channel_size(dataset)
    dataset = dataset.prefetch(FLAGS.prefetch)
    dataset = dataset.batch(1)
    return dataset


def generator_from_dataset(ds, sess):
    """
    this function creates generator object from dataset
    Args:
        dataset: dataset
        sess: tf.Session for evaluate tensors
    """
    iterator = ds.make_one_shot_iterator()
    next_row = iterator.get_next()
    try:
        while True:
            yield sess.run(next_row)

    except tf.errors.OutOfRangeError:
        pass


def generate_tf_record(path, ds, parse_fn=None):
    """
    this function generates tfrecords from dataset
    Args:
        path: direcotyr for tfrecords
        ds: dataset from which tfrecords are created
        parse_fn: parse function to generate example
    Returns:
        None
    """

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _int64_feature_list(values):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

    def _bytes_feature_list(values):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))

    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _float_feature_list(values):
        return tf.train.Feature(float_list=tf.train.FloatList(value=values))

    def row_to_example(row):
        """this function convert row to example"""
        feature_dict = row[0]
        label = row[1]

        return tf.train.Example(
            features=tf.train.Features(
                feature={
                    "word": _bytes_feature_list(feature_dict["word"]),
                    "label": _int64_feature_list(label),
                }
            )
        )

    if parse_fn is None:
        parse_fn = row_to_example

    config_session = tf.ConfigProto(
        inter_op_parallelism_threads=FLAGS.cores,
        intra_op_parallelism_threads=FLAGS.cores,
    )

    counter = 0
    with tf.Session(config=config_session) as sess, tf.python_io.TFRecordWriter(
        path
    ) as writer:
        for row in generator_from_dataset(ds, sess):
            counter += 1
            print("[Sentence Analysys] NOW:" + str(counter))
            example = parse_fn(row)
            writer.write(example.SerializeToString())


class FLAGS:
    cores = multiprocessing.cpu_count()
    prefetch = 1
    batch_size = 2
    init_image_size = 512
    intermediate_image_size = 180
    final_image_size = intermediate_image_size
    shuffle_buffer_size = 50 * batch_size

    gpus = len(util.get_available_gpus())
    if gpus != 0:
        batch_size = int(batch_size / gpus)
        print('INFO: batch size per GPU is {}'.format(batch_size))
        print()
