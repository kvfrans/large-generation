import numpy as np
import tensorflow as tf

import time
import os
import cPickle

from mnist_data import *
from main import Model

'''
cppn vae:

compositional pattern-producing generative adversarial network

LOADS of help was taken from:

https://github.com/carpedm20/DCGAN-tensorflow
https://jmetzen.github.io/2015-11-27/vae.html

'''

def main():

  return train()

def train():

  mnist = read_data_sets()
  n_samples = mnist.num_examples

  cppnvae = Model()

  counter = 0

  # Training cycle
  for epoch in range(100):
    avg_d_loss = 0.
    avg_q_loss = 0.
    avg_vae_loss = 0.
    mnist.shuffle_data()
    total_batch = int(n_samples / 64)
    # Loop over all batches
    for i in range(total_batch):
      batch_images = mnist.next_batch(64)

      d_loss, g_loss, vae_loss = cppnvae.train(batch_images)

    #   assert( vae_loss < 1000000 ) # make sure it is not NaN or Inf
    #   assert( d_loss < 1000000 ) # make sure it is not NaN or Inf
    #   assert( g_loss < 1000000 ) # make sure it is not NaN or Inf

      # Display logs per epoch step
      if (counter+1) % 1 == 0:
        print "Sample:", '%d' % ((i+1)*64), " Epoch:", '%d' % (epoch), \
              "d_loss=", "{:.4f}".format(d_loss), \
              "g_loss=", "{:.4f}".format(g_loss), \
              "vae_loss=", "{:.4f}".format(vae_loss)
      counter += 1

if __name__ == '__main__':
  main()
