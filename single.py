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
        self.beta1 = 0.9
        self.learningrate_d = 0.001
        self.learningrate_g = 0.01


        self.images = tf.placeholder(tf.float32,[self.batchsize,self.xdim,self.ydim,self.cdim])
        self.npoints = self.xdim*self.ydim
        # self.xvec, self.yvec = self.coordinates(self.xdim,self.ydim,self.scale)

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
        # self.G = self.generator_absolute()
        self.G = self.generator()

        self.gloss = tf.nn.l2_loss(self.images - self.G)

        self.g_optim = tf.train.AdamOptimizer(self.learningrate_g, beta1=self.beta1).minimize(self.gloss)

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
        x_unroll = tf.reshape(self.x, [self.batchsize*self.npoints, 1])
        y_unroll = tf.reshape(self.y, [self.batchsize*self.npoints, 1])

        U = fully_connected(x_unroll, self.size, 'g_x', with_bias = False) + \
        fully_connected(y_unroll, self.size, 'g_y', with_bias = False)

        H = tf.nn.softplus(U)
        h2 = tf.nn.tanh(fully_connected(H, self.size, 'g_2'))
        h3 = tf.nn.tanh(fully_connected(h2, self.size, 'g_3'))
        h4 = tf.nn.tanh(fully_connected(h3, self.size, 'g_4'))
        output = fully_connected(h4, self.cdim, 'g_final')
        result = tf.reshape(output, [self.batchsize, self.ydim, self.xdim, self.cdim])
        return tf.nn.tanh(result)

    def generator_absolute(self):
        z = tf.ones((self.batchsize,2), dtype=tf.float32)
        z2 = dense(z, 2, 7*7*self.df_dim*4, scope='g_h0_lin')
        h0 = tf.nn.relu(self.g_bn0(tf.reshape(z2, [-1, 7, 7, self.df_dim*4]))) # 4x4x256
        h1 = tf.nn.relu(self.g_bn1(conv_transpose(h0, [self.batchsize, 14, 14, self.df_dim*2], "g_h1"))) #8x8x128
        h4 = conv_transpose(h1, [self.batchsize, 28, 28, 1], "g_h4")
        return tf.nn.tanh(h4)

    def generate(self, z=None):
        x_vec, y_vec = self.coordinates(self.xdim, self.ydim, self.scale)
        image = self.sess.run(self.G, feed_dict={self.x: x_vec, self.y: y_vec})
        return image


    def train(self):
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        n_samples = mnist.train.num_examples
        x_vec, y_vec = self.coordinates(self.xdim, self.ydim, self.scale)

        troll = np.zeros((28,28))
        troll[:,:14] = 1
        overfit = np.repeat(mnist.train.next_batch(1)[0].reshape(1,28,28,1),64,0)
        # overfit = np.repeat(troll.reshape(1,28,28,1),64,0)
        closs_d = closs_g = 0.65

        for epoch in range(10):
            for idx in range(int(n_samples / self.batchsize)):
                # batch = mnist.train.next_batch(self.batchsize)[0].reshape(self.batchsize,28,28,1)
                batch = overfit
                for i in xrange(1):
                    closs_g, _ = self.sess.run([self.gloss, self.g_optim],feed_dict={ self.x: x_vec, self.y: y_vec, self.images: batch})
                print "%d d: %f g: %f" % (idx, closs_d, closs_g)
                if idx % 50 == 0:
                    print self.generate()[0,:,:,0]
                    ims("results1/"+str(idx)+".jpg",self.generate()[0,:,:,0])
                    ims("results1/o.jpg",troll)



model = Model()
model.train()
