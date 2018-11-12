#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os
import argparse
import matplotlib.pyplot as plt
import scipy.misc
import numpy as np
import tensorflow as tf

from sampler import Sampler
from model import ACGAN
from helper import view_batch


def show_examples(acgan, labels):
    for l,n in enumerate(labels):
        y = acgan.sampler.one_hot(l)

        z = acgan.sampler.z_sampler()
        x = acgan.sess.run(acgan.g, feed_dict={ acgan.z: z, acgan.y: y })
        d_ = acgan.sess.run(acgan.d, feed_dict={ acgan.x: x, acgan.y: y })
        v = view_batch(acgan.sampler.data2img(x))
        plt.figure("generated-%s" % n), plt.imshow(v)
        print "Generated", n, np.mean(d_)

        x = acgan.sampler.x_sampler(l)
        d = acgan.sess.run(acgan.d, feed_dict={ acgan.x: x, acgan.y: y })
        v = view_batch(acgan.sampler.data2img(x))
        plt.figure("real-%s" % n), plt.imshow(v)
        print "Real", n, np.mean(d)
    plt.show()


def save_examples(acgan, labels):
    acgan.sampler.batch_size = 100
    for i in range(10):
      z = acgan.sampler.z_sampler()
      for l,n in enumerate(labels):
        y = acgan.sampler.one_hot(l)

        path = os.path.join("export",n)
        if not os.path.exists(path):
            os.makedirs(path)
        res = acgan.sess.run(acgan.g,feed_dict={ acgan.z: z , acgan.y: y })
        for bi,r in enumerate(res):
          scipy.misc.imsave(os.path.join(path,"%03d.jpg" %
              (i*acgan.sampler.batch_size+bi)), r)


def show_filters():
    with tf.variable_scope("discriminator", reuse=True):
        filters = acgan.sess.run(tf.get_variable("Conv/weights"))
        filters = filters.transpose([2,0,1,3])
        print filters.shape
        v = view_batch(filters).transpose([1,2,0])
        plt.imshow(v, interpolation='none')
        plt.show()


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 0-3

    parser = argparse.ArgumentParser('')
    parser.add_argument('--load', type=str)
    parser.add_argument('--train', action="store_true", default=False)
    args = parser.parse_args()

    sampler = Sampler(path="data", categories=["imagenet256", "landscape256"],
                batch_size=64)
    acgan = ACGAN(sampler, z_dim=100, load=(args.load if args.load else None))

    if args.train:
        acgan.train()
    else:
        if False: show_examples(acgan, ("imagenet","impress"))
        if True: save_examples(acgan, ["imagenet256", "landscape256"])
        if False: show_filters()
