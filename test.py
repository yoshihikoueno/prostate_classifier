#!/usr/bin/python

import unet
import engine

def main():
    '''main'''
    engine.train(unet, 'annotation', steps=10000000)
    return

if __name__ == '__main__':
    main()
