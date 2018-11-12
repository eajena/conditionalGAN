#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def view_batch(batch):
    def grid(num):
        a = int(num/np.sqrt(num))
        for i in range(a, 0, -1):
            if num%i==0: return i, num/i

    num, h, w, c = batch.shape
    a, b = grid(num) 
    v = np.reshape(batch, [a, b, h, w, c])
    v = np.transpose(v, [0, 2, 1, 3, 4])
    v = np.reshape(v, [a * h, b * w, c])
    return v
