# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 16:26:27 2018

@author: arden
"""

import glob
import os

def generate_label_file(img_dir, label_file, img_ext='jpg'):
    cwd = os.getcwd()
    os.chdir(img_dir)
    imgs = glob.glob('*' + img_ext)
    
    labels = list('ABCDEFGHIKLMNOPQRSTUVWXY')
    
    with open(label_file, 'a') as f:
        for img in imgs:
            label = labels.index(img[0])
            outstr = img
            for j in range(24):
                if label == j:
                    outstr += ' 1.0'
                else:
                    outstr += ' 0.0'
            outstr += '\n'
            f.write(outstr)
    os.chdir(cwd)
    
if __name__=="__main__":
    generate_label_file('user_3','user_3.txt')