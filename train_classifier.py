#!/usr/bin/python

import model_annotator
import engine

def main():
    '''main'''
    engine.train(model_annotator, 'both', steps=10000000, no_healthy=True, model_dir='summary_classifier')
    return

if __name__ == '__main__':
    main()
