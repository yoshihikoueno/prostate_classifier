#!/usr/bin/python

import model_classifier_cheat_binary
import engine
import tensorflow as tf

def main():
    '''main'''
    engine.train(
        model_classifier_cheat_binary, 'both', steps=10000000, no_healthy=True,
        model_dir='summary/summary_classifier_binary',
    )
    return

if __name__ == '__main__':
    main()
