import tensorflow as tf
import numpy as np
from ops import *
from scipy.misc import imsave as ims
import input_data

class Model():
    def __init__(self):
        self.batchsize = 64
        self.imgdim = 28
        self.zdim = 32
        self.cdim = 1
        self.df_dim = 24
        self.beta1 = 0.65
        self.learningrate_d = 0.001
        self.learningrate_g = 0.001

        self.images = tf.placeholder(tf.float32,[self.batchsize,self.imgdim,self.imgdim,self.cdim])

        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')

        self.z = tf.placeholder(tf.float32, [self.batchsize, self.zdim])

        # generate images from self.z,
        self.G = self.generator_absolute()

        # try to discriminate real images as 1
        D_prob = self.discriminator(self.images, reuse=False)
        # try to discriminate fake images as 0
        D_fake_prob = self.discriminator(self.G, reuse=True)

        self.d_loss_real = binary_cross_entropy_with_logits(tf.ones_like(D_prob), D_prob)
        self.d_loss_fake = binary_cross_entropy_with_logits(tf.zeros_like(D_fake_prob), D_fake_prob)
        self.dloss = 1.0*(self.d_loss_real + self.d_loss_fake)/ 2.0
        self.gloss = 1.0*binary_cross_entropy_with_logits(tf.ones_like(D_fake_prob), D_fake_prob)

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        self.d_optim = tf.train.AdamOptimizer(self.learningrate_d, beta1=self.beta1).minimize(self.dloss, var_list=d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.learningrate_g, beta1=self.beta1).minimize(self.gloss, var_list=g_vars)

        init = tf.initialize_all_variables()
        # Launch the session
        self.sess = tf.Session()
        self.sess.run(init)

    def generator_absolute(self):
        z2 = dense(self.z, self.zdim, 7*7*self.df_dim*4, scope='g_h0_lin')
        h0 = tf.nn.relu(self.g_bn0(tf.reshape(z2, [-1, 7, 7, self.df_dim*4]))) # 4x4x256
        h1 = tf.nn.relu(self.g_bn1(conv_transpose(h0, [self.batchsize, 14, 14, self.df_dim*2], "g_h1"))) #8x8x128
        h4 = conv_transpose(h1, [self.batchsize, 28, 28, 1], "g_h4")
        return tf.nn.sigmoid(h4)

    def generate(self, z=None):
        if z is None:
            z = np.random.uniform(-1.0,1.0, size=(self.batchsize, self.zdim)).astype(np.float32)

        x_vec, y_vec, r_vec = self.coordinates(self.xdim, self.ydim, self.scale)
        image = self.sess.run(self.G, feed_dict={self.z: z, self.x: x_vec, self.y: y_vec, self.r: r_vec})
        return image

    def discriminator(self, image, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        # this is for MNIST
        # it starts at 28x28x1
        h0 = lrelu(conv2d(image, 1, self.df_dim, name='d_h0_conv')) #14x14x64
        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim, self.df_dim*2, name='d_h1_conv'))) #7x7x128
        h4 = dense(tf.reshape(h1, [self.batchsize, -1]), 7*7*self.df_dim*2, 1, scope='d_h3_lin')
        return tf.nn.sigmoid(h4)

    def train(self):
        closs_d = closs_g = closs_vae = 0.5
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        n_samples = mnist.train.num_examples
        for epoch in range(10):
            for idx in range(int(n_samples / self.batchsize)):
                z = np.random.uniform(-1.0,1.0, size=(self.batchsize, self.zdim)).astype(np.float32)
                batch = mnist.train.next_batch(self.batchsize)[0].reshape([self.batchsize,self.imgdim,self.imgdim,1])
                for i in xrange(4):
                    closs_g, _ = self.sess.run([self.gloss, self.g_optim],feed_dict={self.z: z})

                closs_d, _ = self.sess.run([self.dloss, self.d_optim],feed_dict={self.z: z, self.images: batch})

                print "%d: %f %f" % (idx, closs_d, closs_g)

                if idx % 15 == 0:
                    generated_test = self.sess.run(self.G, feed_dict={self.z: z})
                    generated_test = generated_test.reshape(self.batchsize,28,28)
                    ims("results/"+str(idx + epoch*100000)+".jpg",merge(generated_test[:64],[8,8]))

model = Model()
model.train()
