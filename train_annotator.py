#!/usr/bin/python

import model_annotator
import engine

def main():
    '''main'''
    engine.train(
        model_annotator, 'annotation', steps=10000000, no_healthy=False,
        model_dir='summary/summary_annotator'
    )
    return

if __name__ == '__main__':
    main()
