#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.layers as tcl

from datetime import datetime
from helper import view_batch


class ACGAN(object):
  def __init__(self, sampler, z_dim=100, load=None):
      self.exp_name = "acgan-%s" % (datetime.now().strftime("%Y%m%d-%H%M%S"))
      self.sampler = sampler
      self.z_dim, self.y_dim = z_dim, 2
      self.sampler.z_dim = self.z_dim
      self.sampler.y_dim = self.y_dim

      self.y = tf.placeholder(tf.float32, [None, self.y_dim], name='y')
      self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
      self.x = tf.placeholder(tf.float32, [None, 256, 256, 3], name='x')

      self.g  = self.generator(self.z, self.y)
      self.d  = self.discriminator(self.x, self.y)
      self.d_ = self.discriminator(self.g, self.y, reuse=True)

      t_vars = tf.trainable_variables()
      self.d_vars = [v for v in t_vars if 'discriminator' in v.name]
      self.g_vars = [v for v in t_vars if 'generator' in v.name]
      
      self.loss_g = tf.reduce_mean(self.d_)
      self.loss_d = tf.reduce_mean(self.d) - tf.reduce_mean(self.d_)

      #def L2(a, b): return tf.reduce_mean(tf.square(a - b))
      #self.loss_g = L2(self.d_, 0)
      #self.loss_d = L2(self.d,  0) + L2(self.d_, 1)*0.1

      #-- improved wgan, taken from github.com/jiamings/wgan
      #epsilon = tf.random_uniform([], 0.0, 1.0)
      #x_hat = epsilon * self.x + (1 - epsilon) * self.g
      #d_hat = self.discriminator(x_hat, self.y, reuse=True)
      #ddx = tf.gradients(d_hat, x_hat)[0]
      #ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1))
      #ddx = tf.reduce_mean(tf.square(ddx - 1.0) * 10)
      #self.loss_d = self.loss_d + ddx
      #--

      with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
          self.d_optim = tf.train.AdamOptimizer(
                            learning_rate=1e-4, beta1=0.5, beta2=0.9
                            ).minimize(self.loss_d, var_list=self.d_vars)
          self.g_optim = tf.train.AdamOptimizer(
                            learning_rate=1e-4, beta1=0.5, beta2=0.9
                            ).minimize(self.loss_g, var_list=self.g_vars)

      #self.d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in self.d_vars]

      config = tf.ConfigProto(
          device_count = {'GPU': 1 if load is None else 0},
          gpu_options=tf.GPUOptions(allow_growth=True)
      )
      self.sess = tf.Session(config=config)
      self.saver = tf.train.Saver()

      if load is None:
          self.sess.run(tf.global_variables_initializer())
      else:
          self.load(load)

      tf.summary.scalar("loss_d", self.loss_d)
      tf.summary.scalar("loss_g", self.loss_g)
      tf.summary.histogram("self.d", self.d)
      tf.summary.histogram("self.g", self.g)
      self.summaries = tf.summary.merge_all()


  def load(self, path):
      self.checkpoints = path
      self.saver.restore(self.sess, tf.train.latest_checkpoint(path))

  
  def train(self, epochs=100000):
      self.summary_writer = tf.summary.FileWriter('logs/'+self.exp_name,
              self.sess.graph)
      test_z = self.sampler.z_sampler()
      test_y = self.sampler.one_hot(0), self.sampler.one_hot(1)

      for t in range(epochs):
        for label in [0,1]:
          y = self.sampler.one_hot(label)
          for i in range(1):
              x = self.sampler.x_sampler(label)
              z = self.sampler.z_sampler()
              #self.sess.run(self.d_clip)
              self.sess.run(self.d_optim,
                      feed_dict={self.x: x, self.z: z, self.y: y})

          z = self.sampler.z_sampler()
          _, summary = self.sess.run([self.g_optim, self.summaries],
                    feed_dict={self.x: x, self.z: z, self.y: y})
          self.summary_writer.add_summary(summary, t)

          if t % 100 == 0:
              test_x = self.sampler.x_sampler(label)
              loss_d, loss_g, img = self.sess.run(
                  [self.loss_d, self.loss_g, self.g],
                  feed_dict={self.x: test_x, self.z: test_z, self.y: test_y[label]})
              print('%8d-%d:  d_loss %.4f  g_loss %.4f' %
                      (t, label, loss_d, loss_g))

              img = view_batch(self.sampler.data2img(img))
              if not os.path.exists("samples/%s" % (self.exp_name)):
                  os.makedirs("samples/%s" % (self.exp_name))
              scipy.misc.imsave(
                  'samples/%s/%d-%d.jpg' % (self.exp_name, t/100, label), img)
          if t % 1000 == 0:
              path = os.path.join("checkpoints", self.exp_name)
              if not os.path.exists(path):
                  os.makedirs(path)
              self.saver.save(self.sess, os.path.join(path, self.exp_name),
                  global_step=t)


  def discriminator(self, x, y, reuse=False):
      with tf.variable_scope("discriminator") as scope:
          if reuse: scope.reuse_variables()

          x = tf.reshape(x, [tf.shape(x)[0], 256, 256, 3])
          self.conv1 = tcl.conv2d(
              x, 16, kernel_size=[3, 3], stride=[2, 2],
              weights_initializer=tf.random_normal_initializer(stddev=0.02),
              activation_fn=lrelu_batch_norm
          )
          self.conv2 = tcl.conv2d(
              self.conv1, 32, kernel_size=[3, 3], stride=[2, 2],
              weights_initializer=tf.random_normal_initializer(stddev=0.02),
              activation_fn=lrelu_batch_norm
          )
          self.conv3 = tcl.conv2d(
              self.conv2, 64, kernel_size=[3, 3], stride=[2, 2],
              weights_initializer=tf.random_normal_initializer(stddev=0.02),
              activation_fn=lrelu_batch_norm
          )
          self.conv4 = tcl.conv2d(
              self.conv3, 128, kernel_size=[3, 3], stride=[1, 1],
              weights_initializer=tf.random_normal_initializer(stddev=0.02),
              activation_fn=lrelu_batch_norm
          )
          self.conv5 = tcl.conv2d(
              self.conv4, 256, kernel_size=[3, 3], stride=[2, 2],
              weights_initializer=tf.random_normal_initializer(stddev=0.02),
              activation_fn=lrelu_batch_norm
          )
          self.conv6 = tcl.conv2d(
              self.conv5, 512, kernel_size=[3, 3], stride=[1, 1],
              weights_initializer=tf.random_normal_initializer(stddev=0.02),
              activation_fn=lrelu_batch_norm
          )
          self.fc = tcl.fully_connected(
              tf.concat([tcl.flatten(self.conv6), y], axis=1), 1,
              activation_fn=tf.identity
          )

      return self.fc


  def generator(self, z, y):
      with tf.variable_scope("generator") as scope:

          batch_size = tf.shape(z)[0]
          self.fc = tcl.fully_connected(
              tf.concat([z, y], axis=1), 768,
              activation_fn=tf.nn.relu
          )
          self.conv1 = tcl.conv2d_transpose(
              tf.reshape(self.fc, [batch_size, 4, 4, 48]),
              384, kernel_size=[5, 5], stride=[2, 2],
              weights_initializer=tf.random_normal_initializer(stddev=0.02),
              activation_fn=relu_batch_norm
          )
          self.conv2 = tcl.conv2d_transpose(
              self.conv1, 256, kernel_size=[5, 5], stride=[4, 4],
              weights_initializer=tf.random_normal_initializer(stddev=0.02),
              activation_fn=relu_batch_norm
          )
          self.conv3 = tcl.conv2d_transpose(
              self.conv2, 192, kernel_size=[5, 5], stride=[2, 2],
              weights_initializer=tf.random_normal_initializer(stddev=0.02),
              activation_fn=relu_batch_norm
          )
          self.conv4 = tcl.conv2d_transpose(
              self.conv3, 64, kernel_size=[5, 5], stride=[2, 2],
              weights_initializer=tf.random_normal_initializer(stddev=0.02),
              activation_fn=relu_batch_norm
          )
          self.conv5 = tcl.conv2d_transpose(
              self.conv4, 3, kernel_size=[5, 5], stride=[2, 2],
              weights_initializer=tf.random_normal_initializer(stddev=0.02),
              activation_fn=tf.tanh
          )
          return self.conv5


def relu(x):
    return tf.nn.relu(x)

def relu_batch_norm(x):
    return tf.nn.relu(tcl.batch_norm(x))

def lrelu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)

def lrelu_batch_norm(x, alpha=0.2):
    return lrelu(tcl.batch_norm(x), alpha)
