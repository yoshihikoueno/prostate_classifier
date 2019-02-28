"""
This module contains some utilities for PDF/TXT analysis.
"""
import random
from operator import add
import regex
import numpy as np
import urllib
import tensorflow as tf
from functools import reduce
from tensorflow.python.client import device_lib
import os
import multiprocessing as mp
import cv2
import wget
import pandas as pd
import matplotlib.pyplot as plt
label_font_size = 15
label_value = (255, 0, 0)

def extract_label(path):
    '''
    this func will decode anno image and extract label from it
    '''
    image = cv2.imread(path)
    label = (image == label_value)
    return label

def decode_image(path):
    '''
    this func will decode the specified image and return a dict
    '''
    result = dict()
    result['path'] = path
    result['image'] = cv2.imread(path)
    return result

def find_dirs(dir_):
    '''
    this func will return a list strings which represent
    paths for the directory located directory under 'dir_'
    '''
    dirs = find_children(dir_)
    dirs = list(filter(os.path.isdir, dirs))
    return dirs

def find_files(dir_):
    '''
    this func will return a list strings which represent
    paths for the directory located directory under 'dir_'
    '''
    dirs = find_children(dir_)
    dirs = list(filter(os.path.isfile, dirs))
    return dirs

def find_children(dir_):
    '''
    this func will return a list strings which represent
    paths for the objects located directory under 'dir_'
    '''
    dirs = os.listdir(dir_)
    dirs = list(map(lambda x: concatenate_dir(dir_, x), dirs))
    return dirs

def concatenate_dir(d1, d2):
    '''
    this func will concatename 2 names,
    both of them represents name of a directory or a file
    '''
    if d1[-1] == '/':
        d1 = d1[:-1]
    return '{}/{}'.format(d1, d2)

def get_anno_raw(root_dir='data/RAW/cancer_annotations', decode=False, parallel=True):
    '''
    this func will return a list of
    dicts{'annotation': {'path': [str], 'image': [np.array]}, 'raw': {'path':[str], 'image': [np.array]}}
    '''
    anno = get_all_annotations(root_dir)
    raw = list(map(get_raw, anno))
    if decode:
        if parallel:
            with mp.Pool(mp.cpu_count()) as p:
                anno = p.map(decode_image, anno)
                raw = p.map(decode_image, raw)
        else:
            anno = list(map(decode_image, anno))
            raw = list(map(decode_image, raw))
    anno_raw = list(zip(anno, raw))
    dicts = list(map(lambda x: {'annotation': x[0], 'raw': x[1]}, anno_raw))
    return dicts

def get_all_annotations(root_dir='data/RAW/cancer_annotations'):
    '''
    this func will return the list of strings which represent
    paths for the annotations.

    Args:
        root_dir:[optional] the directory where this func will look up annotations
    '''
    files = []
    new_dirs = [root_dir]
    while new_dirs:
        new_files = reduce(add, list(map(find_files, new_dirs)))
        new_dirs = reduce(add, list(map(find_dirs, new_dirs)))
        files += new_files
    return files

def get_csv(tag_list, start_dir, host_name='localhost', port=6006, save_dir='.', plot_combined_eval=True, combine=True):
    '''
    thsi function will download summaries for specified
    tags and for every findable runs
    NOTE: require tensorboard to be ready
    '''
    csv_dir = '{}/csv'.format(save_dir)
    figure_dir = '{}/figure'.format(save_dir)

    for tag in tag_list:
        get_CSV_tag(tag, start_dir, csv_dir, host_name, port)

    if plot_combined_eval:
        plot_combined_eval_summary(csv_dir, figure_dir)
    if combine:
        combine_summary_csv(csv_dir)
    plot_summary_csv(csv_dir, figure_dir)
    return

def plot_combined_eval_summary(csv_dir, save_dir):
    '''
    this func will plot combined figure for each metric
    of all the evaluations
    '''
    name_conversion = {
        'classifier_cheat_bn_proper': 4,
        'classifier_cheat': 2,
        'classifier_cheat_new': 3,
        'classifier': 1,
    }
    exclude_list = [
        'classifier_cheat_bn',
        'classifier_cheat_bn_smaller',
        'classifier_cheat_bn_warp',
    ]
    files = os.listdir(csv_dir)
    files = list(filter(lambda x: 'annotator' not in x, files))
    files = list(filter(lambda x: 'binary' not in x, files))
    files = list(map(lambda x: '{}/{}'.format(csv_dir, x), files))
    files = list(filter(lambda file_: 'Evaluation' in file_, files))

    tag_list = set(map(lambda file_: get_run_mode_tag(file_)[-1], files))

    for tag in tag_list:
        combine_targets = list(filter(lambda file_: get_run_mode_tag(file_)[-1] == tag, files))
        run_df_list = list(map(lambda file_: (get_run_mode_tag(file_)[0], pd.read_csv(file_, index_col=0)), combine_targets))
        series_list = []
        for run, df in run_df_list:
            series = df.iloc[:, 0].dropna()
            series.name = run
            series.name = regex.sub('summary_', '', series.name)
            if series.name in exclude_list:
                continue
            if series.name in name_conversion.keys():
                series.name = name_conversion[series.name]
            series_list.append(series)
        print('INFO: found {} evaluation series for {}'.format(len(series_list), tag))
        series_list = sorted(series_list, key=lambda x: x.name)
        for series in series_list:
            series.name = 'EXP{}'.format(series.name)

        if tag == 'accuracy':
            vars_ = list(map(lambda s: s.max(), series_list))
        elif tag == 'loss':
            vars_ = list(map(lambda s: s.min(), series_list))
        else:
            vars_ = list(map(lambda s: s.min(), series_list))
        index_list = np.arange(len(vars_)) * 2
        name_list = list(map(lambda s: '{}'.format(s.name, s.argmax()), series_list))
        # name_list = list(map(lambda s: '{}\n{}'.format(s.name, s.argmax()), series_list))
        _, ax = plt.subplots()
        plt.bar(index_list, vars_)
        plt.xticks(index_list, name_list)
        ax.set_ylabel(tag)
        ax.set_xlabel('configuration')
        ax.yaxis.label.set_size(label_font_size)
        ax.xaxis.label.set_size(label_font_size)
        plt.tight_layout()
        plt.savefig('{}/{}_bar.png'.format(save_dir, tag))
        plt.close()

        min_index = min(list(map(lambda s: s.index[-1], series_list)))
        series_list = list(map(lambda s: s.loc[:min_index], series_list))
        _, ax = plt.subplots()
        for series in series_list:
            series.plot(legend=True).set_ylabel(tag)
        plt.tight_layout()
        ax.yaxis.label.set_size(label_font_size)
        ax.xaxis.label.set_size(label_font_size)
        plt.savefig('{}/{}.png'.format(save_dir, tag))
        plt.close()

    return

def get_CSV_tag(tag, start_dir, save_dir=None, host_name='localhost', port=6006):
    '''
    this func will download CSV files from
    tensorboard server for all the runs
    NOTE: require tensorboard to be ready
    '''
    runs = find_runs(start_dir)
    for run in runs:
        mode = 'Evaluation' if 'eval' in run else 'Training'
        run_raw = run
        run = regex.sub(r'/', '_', run)
        run = regex.sub('_eval', '', run)
        if save_dir is not None:
            file_name = '{}/{}.csv'.format(save_dir, '_'.join([run, mode, tag]))
        get_csv_from_tensorboard(tag, run_raw, file_name, host_name, port)
    return

def get_run_tag(file_name):
    '''
    this func will extract run_name, mode and tag
    '''
    file_name = regex.sub(r'.*/', '', file_name)
    file_name = file_name[:-4]

    *run, tag = file_name.split('_')
    run_name = '_'.join(run)
    return run_name, tag

def get_run_mode_tag(file_name):
    '''
    this func will extract run_name, mode and tag
    '''
    file_name = regex.sub(r'.*/', '', file_name)
    file_name = file_name[:-4]

    run_name = regex.sub(r'(.*)_(Evaluation|Training)_(.*)', r'\1', file_name)
    mode = regex.sub(r'(.*)_(Evaluation|Training)_(.*)', r'\2', file_name)
    tag = regex.sub(r'(.*)_(Evaluation|Training)_(.*)', r'\3', file_name)
    return run_name, mode, tag

def plot_summary_csv(csv_dir, save_dir, overwrite=True):
    '''
    this func will make figures for all the CSV files
    '''
    if save_dir[-1] == '/':
        save_dir = save_dir[:-1]

    files = os.listdir(csv_dir)
    files = list(filter(lambda x: '.csv' == x[-4:], files))

    for file_ in files:
        _, ax = plt.subplots()
        if os.path.exists('{}/{}'.format(save_dir, change_extension(file_, 'png'))):
            os.remove('{}/{}'.format(save_dir, change_extension(file_, 'png')))

        _, tag = get_run_tag(file_)
        ylabel = summary_tag_modifier(tag)
        df = pd.read_csv('{}/{}'.format(csv_dir, file_), index_col=0)
        if 'Evaluation' in df.columns and 'Training' in df.columns:
            print('INFO: plotting Eval and Train: {}'.format(file_))
            df.Evaluation.dropna().plot(legend=True).set_ylabel(ylabel)
            df.Training.dropna().plot(legend=True).set_ylabel(ylabel)
        else:
            df.iloc[:, 0].plot(legend=False).set_ylabel(ylabel)
        plt.tight_layout()
        ax.yaxis.label.set_size(label_font_size)
        ax.xaxis.label.set_size(label_font_size)
        plt.savefig('{}/{}'.format(save_dir, change_extension(file_, 'png')))
        plt.close()

def combine_summary_csv(csv_dir, remove_old=True):
    '''
    this function will combine summaries for training and eval
    into single CSV
    '''
    files = os.listdir(csv_dir)
    files = list(filter(lambda x: '.csv' == x[-4:], files))
    files = list(filter(lambda x: 'Evaluation' in x or 'Training' in x, files))
    files = list(map(lambda x: '{}/{}'.format(csv_dir, x), files))
    for file_ in files:
        run_name, mode, tag = get_run_mode_tag(file_)
        eval_file = regex.sub('Training', 'Evaluation', file_)
        combined_file = regex.sub('_Training', '', file_)

        if mode == 'Evaluation':
            print('Skipping evaluation file: {}'.format(file_))
            continue
        if eval_file not in files:
            print('Skipping file without eval: {}'.format(file_))
            continue

        df_eval = pd.read_csv(eval_file, index_col=0)
        df_train = pd.read_csv(file_, index_col=0)

        try:
            df_eval.columns = ['Evaluation']
        except ValueError:
            print("ERROR: ValueError caused in {}".format(eval_file))
            exit(-1)

        try:
            df_train.columns = ['Training']
        except ValueError:
            print("ERROR: ValueError caused in {}".format(file_))
            exit(-1)

        df = pd.concat([df_eval, df_train], axis=1)
        df.to_csv(combined_file)
        print('Combined file: {}'.format(combined_file))
        if remove_old:
            os.remove(file_)
            os.remove(eval_file)
            print('Deleted file: {}'.format(file_))
            print('Deleted file: {}'.format(eval_file))

def modify_summary_csv(file_name, tag, replace=True):
    '''
    this function will put column list in CSV files
    the number of elements in col_list must be exactly
    the same as the number of columns in CSV

    setting replace=True will remove existing the first line
    in the CSV file and put new line for column labels
    '''
    df = pd.read_csv(file_name, index_col=1)
    df = df.drop(df.columns[0], 1)
    if replace:
        df.columns = [tag]
    df.to_csv(file_name)
    return

def summary_tag_modifier(tag):
    '''
    this func will modify tag
    '''
    rule = {
        'seg': 'loss for segmentation',
        'group': 'loss for classification',
    }

    if tag in rule.keys():
        tag = rule[tag]

    return tag

def get_csv_from_tensorboard(
        tag_name,
        run_name,
        file_name,
        host_name='localhost',
        port=6006,
        overwrite=True,
):
    '''
    this function will query CSV request to tensorboard
    NOTE: require tensorboard to be ready
    '''
    if overwrite and os.path.exists(file_name):
        print('WARNING: removing file {} because of overwrite={}'.format(
            file_name, overwrite
        ))
        os.remove(file_name)

    url = 'http://{}:{}/data/plugin/scalars/scalars?tag={}&run={}&experiment=&format=csv'.format(
        host_name, port, tag_name, run_name
    )
    try:
        print('Downloading: {}'.format(url))
        print('File name: {}'.format(file_name))
        print('Tag: {}'.format(tag_name))
        wget.download(url, file_name)
        modify_summary_csv(file_name, tag_name)
    except urllib.error.HTTPError:
        print('No such data')
    print()

def find_runs(start_dir):
    '''
    find all the directory containing summaries
    '''
    if start_dir[-1] == '/':
        start_dir = start_dir[:-1]
    runs = []
    for root, dirs, files in os.walk(start_dir):
        runs_in = list(map(lambda x: '{}/{}'.format(root, x), dirs))
        runs += runs_in

    runs = list(map(lambda x: x[len(start_dir) + 1:], runs))
    return runs

def find_event_files(start_dir):
    '''
    find all the tf.event files
    '''
    events = []
    for root, dirs, files in os.walk(start_dir):
        events_inner = list(filter(lambda x: 'events' in x, files))
        events_inner = list(map(lambda x: '{}/{}'.format(root, x), events_inner))
        events += events_inner
    return events

def is_iterable(subject):
    '''
    this func will check if given object
    is iterable or not
    '''
    try:
        it = iter(subject)
    except TypeError:
        return False
    else:
        return True

def tf_threshold(input_tensor, threshold, max_val, dtype=tf.int64):
    '''
    this function will performs thresholding operation
    for input tensor. elements with larger or equal to
    threshold will be max_val otherwise 0
    '''
    output = tf.cast(tf.greater_equal(input_tensor, threshold), dtype)*max_val
    output = tf.cast(output, dtype)
    return output

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def get_config_from_modeldir(model_dir):
    '''
    this funtion will try to get config dict
    from given model directory.
    '''
    if not is_config_available(model_dir):
        return None

    model_dir = model_dir+'/' if model_dir[-1]!='/' else model_dir
    with open(model_dir+'config') as f:
        config_str = f.read()
    params = util.string_to_config(config_str)
    return params

def is_config_available(target_dir):
    """
    this function checks if there is a config file in
    the specified directory.
    """
    return os.path.exists(target_dir) and "config" in os.listdir(target_dir)

def config_validator(config_test, config_ground_truth):
    """
    this function checks if the given config is valid or not
    using config_ground_truth
    """
    return isinstance(config_test, dict) and (config_test.keys() == config_ground_truth.keys())

def config_to_string(config):
    """
    this function converts config to string
    so that it can be used as a directory name or file name
    """
    return regex.sub('[\'\{\}\s]','',str(config))

def string_to_config(string):
    """
    this function converts file_name which is
    generated by "config_to_file_name" func to
    back to config dictionary
    """
    keys = regex.findall(r'[a-zA-Z_]+(?=:)', string)
    keys_vals = list(map(lambda key: (key, regex.sub(r'^.*{}:(.+?)(,.*)?$'.format(key), r'\1', string)), keys))
    keys_vals = list(map(lambda key_val: (key_val[0], ast.literal_eval(key_val[1])), keys_vals))
    return dict(keys_vals)

def get_raw(path_annotation, keyword='cancer_cases', new_extension='jpg'):
    """
    this function will convert string path
    to annotated images to non-annotated images.
    note that should be a directory with name specified
    by keyword parameter. And that directory must have
    exactly the same directory structures and names inside
    as train or eval directory.

    Args:
        path_annotation: string path to
            annotated images
        keyword: (optional) the name of directory which
            contains images for non-annotated images
        new_extension: (optional) if you want to change
            the extension of a file, you can specify the
            new extension to this param
            
    Returns:
        string path to non-annotated images
    """
    if not isinstance(path_annotation, str):
        path_annotation = path_annotation.decode()

    # if given str is a path for healthy cases, we don't
    #  to get extra image
    if 'healthy' in path_annotation:
        return path_annotation

    path_non_annotated = regex.sub(r'((train|eval)/.*?/.*?/)|cancer_annotations/', 'cancer_cases/', path_annotation)
    # making sure that there is a difference
    assert path_non_annotated != path_annotation, 'failed to obtain non-annoated image: {}'.format(path_annotation)
    path_non_annotated = change_extension(path_non_annotated, new_extension)
    return path_non_annotated

def tf_exists(target_file_list):
    '''
    this function will check if all the
    files in target_file_list exist
    this function will return True only
    if ALL the files exist
    '''
    def py_exists(targets):
        '''
        this function actually
        performs the operation
        '''
        for target in targets:
            if not tf.gfile.Exists(target):
                print('WARNING: {} does not exists'.format(target))
                print('WARNING: list : {}\n'.format(targets))
                return False
        return True

    return tf.py_func(
        py_exists, [target_file_list], tf.bool
    )

def tf_get_raw(path_annotation):
    """
    wrapper function for tensorflow
    """
    return tf.py_func(
        get_raw, [path_annotation], tf.string
    )

def tf_determine_image_channel(image, channel_size):
    '''
    this function will determine the
    chennel of a image, which is necessary
    when you are using conv, since they
    require channel size to initialize
    their filter
    Args:
        image: input image
        channel_size: channel size
            3 for 3 channel image
            1 for mono-chromatic image
            other for custom images
    Return:
        image with channel size determined
    '''
    new_shape = list(image.get_shape())

    new_shape[-1] = channel_size
    image.set_shape(new_shape)
    return image

def tf_determine_image_size(image):
    '''
    normally, tf doesn't know the size of
    images and it causes various problems
    sometimes. this developers can use 
    this function to determine them.
    '''
    return tf.image.resize_image_with_crop_or_pad(image, tf.shape(image)[0], tf.shape(image)[1])

def tf_extract_label(image, label_value, gray_scale=True):
    """
    this function extracts label image
    from given image which contains
    background and label on it.
    Args:
        image: input image, must be 1 channel
        label_value: pixel value of label
    Returns:
        image that contains only image
    """
    label_image = tf.cast(tf.equal(image, label_value), image.dtype) * 255

    if gray_scale:
        label_image = label_image[:, :, 0:1]

    return label_image

def config_to_file_name(config):
    """
    this function converts config to string
    so that it can be used as a directory name or file name
    """
    return regex.sub('[\'\{\}\s]','',str(config))

def change_extension(file_name, new_extension):
    """
    this function changes the extension.
    """
    return regex.sub(r'^(.*/)?(.*\.).*$', r'\1\2'+new_extension, file_name)

def get_name(file_name):
    """
    this function extracts name from file name,
    which means getting rid of extension
    """
    name = regex.sub(r'^(.*/)?(.*)\..*$', r'\2', file_name)
    return name

def ceil_div(x, y):
    """
    this function calculates the ceiling of division.
    Args:
        x: integer
        y: integer
    Returns:
        ceil(x/y)
    """
    return (int)((x + y - 1) / y)

def create_cachefile_name(key, extension):
    """
    this function creaates filename for cache.
    Args:
        key: A string from which cache file name is created.
            For example, you can specify pdf file name e.g. '/home/user/Documents/test.pdf'
            and specify 'txt' for extension, in this case, this funciton returns '__cache__test.txt'
            Note that string before the last '/' will be removed. You might want to specify date for key,
            in that case, you should not use date format 'YYYY/MM/DD', instead, use ones like 'YYYY-MM-DD'.

        extension: the extension for the cache file. This can be NULL string, which is dicouraged.

    Returns:
        stirng: cache file name
    """
    return reex.sub(r"(.*/)*(.*\.).*", r"__cache__\2" + extension, key)

def generate_train_test(path, ratio=0.75, path_train='data/train.txt', path_test='data/test.txt'):
    """
    this function generates train data and test data from
     a document specified by 'path'.
    Args:
        path: a path for a document
        ratio: n_train / ( n_test + n_train)
        path_train: a path for train data output
        path_test: a path for test data output
    Returns:
        None
    """
    text = get_text(path)
    text = splitter(text)
    n_train = (int)(len(text) * ratio)

    random.shuffle(text)

    train = text[0:n_train]
    test = text[n_train:len(text)]

    list_to_file(train, path_train)
    list_to_file(test, path_test)

def list_to_file(in_list, file_name):
    """
    this function writes in_list into file named file_name.
    Args:
        in_list: input list
        file_name: file to write
    Returns:
        None
    """
    with open(file_name, "w") as f:
        for s in in_list:
            f.write(s+'\n')

def list_isin(query, list_in):
    """
    this function checks if a list contains some element.
    Args:
        query: the element to check if a list contains.
        list_in: the list where this function check.
    Returns:
        integer: index if it contains, -1 otherwise
    """
    try:
        return list_in.index(query)
    except ValueError:
        return -1
