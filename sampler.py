#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os
import numpy as np
from scipy.ndimage import imread

class Sampler(object):
    def __init__(self, path, categories, batch_size=64):
        self.batch_size = batch_size
        self.path = path
        self.files, self.info, self.pos = {}, {}, {}
        for n,c in enumerate(categories):
            f = os.listdir(os.path.join(path, c))
            self.info[n] = (c, len(f))
            self.files[n] = f
            self.pos[n] = 0


    def x_sampler(self, label):
        files = []
        name, total = self.info[label]
        for i in range(self.batch_size):
            img = imread(os.path.join(self.path, name,
                self.files[label][self.batch_size*self.pos[label]+i]))
            if img.ndim<3:
                img = np.dstack((img,img,img))
            files.append(img)
        self.pos[label] += 1
        if self.pos[label] >= total/self.batch_size:
            self.pos[label] = 0

        return np.asarray(files)/255.*2.-1.


    def data2img(self, data):
        return np.clip((data+1.)/2., 0., 1.)


    def img2data(self, img):
        return img/255.*2.-1.


    def z_sampler(self):
        z = np.random.uniform(-1., 1., [self.batch_size, self.z_dim])
        return z


    def one_hot(self, label):
        y = np.zeros([self.batch_size, self.y_dim])
        y[:,label] = 1.0
        return y
