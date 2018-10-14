"""
This module contains some utilities for PDF/TXT analysis.
"""
import random
import regex
import os

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
