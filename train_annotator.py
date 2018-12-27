#!/usr/bin/python

import unet_cnn
import engine

def main():
    '''main'''
    engine.train(unet_cnn, 'annotation', steps=10000000)
    return

if __name__ == '__main__':
    main()
