import tensorflow as tf
import numpy as np
from ops import *
from scipy.misc import imsave as ims
import input_data

class Model():
    def __init__(self, batchsize=64, zdim=200, cdim = 1, scale=8):
        self.batchsize = batchsize
        self.zdim = zdim
        self.cdim = cdim
        self.scale = scale
        self.size = 32
        self.xdim = 28
        self.ydim = 28
        self.df_dim = 64
        self.beta1 = 0.5
        self.learningrate = 0.0002


        self.images = tf.placeholder(tf.float32,[self.batchsize,self.xdim,self.ydim,self.cdim])
        self.npoints = self.xdim*self.ydim
        # self.xvec, self.yvec = self.coordinates(self.xdim,self.ydim,self.scale)

        self.z = tf.placeholder(tf.float32, [self.batchsize, self.zdim])

        # none = npoints
        self.x = tf.placeholder(tf.float32, [self.batchsize, None, 1])
        self.y = tf.placeholder(tf.float32, [self.batchsize, None, 1])

        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')


        # generate images from self.z, self.xdim, self.ydim
        self.G = self.generator_absolute()

        # try to discriminate real images as 1
        D_prob, D_logit = self.discriminator(self.images, reuse=False)
        # try to discriminate fake images as 0
        D_fake_prob, D_fake_logit = self.discriminator(self.G, reuse=True)

        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_logit, tf.ones_like(D_logit)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_fake_logit, tf.zeros_like(D_fake_logit)))

        self.gloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_fake_logit, tf.ones_like(D_fake_logit)))
        self.dloss = 0.5*d_loss_real + 0.5*d_loss_fake

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        self.d_optim = tf.train.AdamOptimizer(self.learningrate, beta1=self.beta1).minimize(self.dloss, var_list=d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.learningrate, beta1=self.beta1).minimize(self.gloss, var_list=g_vars)

        init = tf.initialize_all_variables()
        # Launch the session
        self.sess = tf.Session()
        self.sess.run(init)

    def coordinates(self, xdim, ydim, scale):
        n_points = xdim * ydim
        x_range = scale*(np.arange(xdim)-(xdim-1)/2.0)/(xdim-1)/0.5
        y_range = scale*(np.arange(ydim)-(ydim-1)/2.0)/(ydim-1)/0.5
        x_mat = np.matmul(np.ones((ydim, 1)), x_range.reshape((1, xdim)))
        y_mat = np.matmul(y_range.reshape((ydim, 1)), np.ones((1, xdim)))
        x_mat = np.tile(x_mat.flatten(), self.batchsize).reshape(self.batchsize, n_points, 1)
        y_mat = np.tile(y_mat.flatten(), self.batchsize).reshape(self.batchsize, n_points, 1)
        return x_mat, y_mat

    def generator(self, reuse = False):
        z_scaled = tf.reshape(self.z, [self.batchsize, 1, self.zdim]) * tf.ones([self.npoints, 1], dtype=tf.float32) * self.scale
        z_unroll = tf.reshape(z_scaled, [self.batchsize*self.npoints, self.zdim])
        x_unroll = tf.reshape(self.x, [self.batchsize*self.npoints, 1])
        y_unroll = tf.reshape(self.y, [self.batchsize*self.npoints, 1])

        U = fully_connected(z_unroll, self.size, 'g_bn0') + \
        fully_connected(x_unroll, self.size, 'g_bn1', with_bias = False) + \
        fully_connected(y_unroll, self.size, 'g_bn2', with_bias = False)

        H = tf.nn.tanh(U)
        for i in range(3):
            H = tf.nn.tanh(fully_connected(H, self.size, 'g_tanh_'+str(i)))
        output = tf.tanh(fully_connected(H, self.cdim, 'g_final'))
        result = tf.reshape(output, [self.batchsize, self.ydim, self.xdim, self.cdim])
        return result

    def generator_absolute(self):
        z2 = dense(self.z, self.zdim, 7*7*self.df_dim*4, scope='g_h0_lin')
        h0 = tf.nn.relu(self.g_bn0(tf.reshape(z2, [-1, 7, 7, self.df_dim*4]))) # 4x4x256
        h1 = tf.nn.relu(self.g_bn1(conv_transpose(h0, [self.batchsize, 14, 14, self.df_dim*2], "g_h1"))) #8x8x128
        h4 = conv_transpose(h1, [self.batchsize, 28, 28, 1], "g_h4")
        return tf.nn.tanh(h4)

    def generate(self, z=None):
        if z is None:
            z = np.random.uniform(-1.0,1.0, size=(self.batchsize, self.zdim)).astype(np.float32)

        x_vec, y_vec = self.coordinates(self.xdim, self.ydim, self.scale)
        image = self.sess.run(self.G, feed_dict={self.z: z, self.x: x_vec, self.y: y_vec})
        return image

    def discriminator(self, image, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        # this is for MNIST
        # it starts at 28x28x1
        h0 = lrelu(conv2d(image, 1, self.df_dim, name='d_h0_conv')) #14x14x64
        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim, self.df_dim*2, name='d_h1_conv'))) #7x7x128
        h4 = dense(tf.reshape(h1, [self.batchsize, -1]), 7*7*self.df_dim*2, 1, scope='d_h3_lin')
        return tf.nn.sigmoid(h4), h4

    def train(self):
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        n_samples = mnist.train.num_examples
        x_vec, y_vec = self.coordinates(self.xdim, self.ydim, self.scale)

        overfit = np.repeat(mnist.train.next_batch(1)[0].reshape(1,28,28,1),64,0)

        for epoch in range(10):
            for idx in range(int(n_samples / self.batchsize)):
                # batch = mnist.train.next_batch(self.batchsize)[0].reshape(self.batchsize,28,28,1)
                batch = overfit
                batch_z = np.random.uniform(-1, 1, [self.batchsize, self.zdim]).astype(np.float32)
                closs_d, _ = self.sess.run([self.dloss, self.d_optim],feed_dict={ self.x: x_vec, self.y: y_vec, self.z: batch_z, self.images: batch})
                for i in xrange(1):
                    closs_g, _ = self.sess.run([self.gloss, self.g_optim],feed_dict={ self.x: x_vec, self.y: y_vec, self.z: batch_z })
                print "%d d: %f g: %f" % (idx, closs_d, closs_g)
                if idx % 5 == 0:
                    ims("results/"+str(idx)+".png",self.generate()[0,:,:,0])
                    ims("results/o.png",overfit[0,:,:,0])



model = Model()
model.train()
