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
        self.discriminator_size = 24
        self.generator_size = 24
        self.encoder_size = 64
        self.beta1 = 0.65
        self.learningrate_d = 0.001
        self.learningrate_g = 0.001
        self.learningrate_vae = 0.001

        self.images = tf.placeholder(tf.float32,[self.batchsize,self.imgdim,self.imgdim,self.cdim])

        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')

        self.z_mean, self.z_stddev = self.encoder(self.images)
        samples = tf.random_normal([self.batchsize,self.zdim],0,1,dtype=tf.float32)
        self.z = self.z_mean + (self.z_stddev * samples)

        # generate images from self.z,
        self.G = self.generator()

        # try to discriminate real images as 1
        D_prob = self.discriminator(self.images, reuse=False)
        # try to discriminate fake images as 0
        D_fake_prob = self.discriminator(self.G, reuse=True)

        self.d_loss_real = binary_cross_entropy_with_logits(tf.ones_like(D_prob), D_prob)
        self.d_loss_fake = binary_cross_entropy_with_logits(tf.zeros_like(D_fake_prob), D_fake_prob)
        self.d_loss = 1.0*(self.d_loss_real + self.d_loss_fake)/ 2.0
        self.g_loss = 1.0*binary_cross_entropy_with_logits(tf.ones_like(D_fake_prob), D_fake_prob)

        flattened_images = tf.reshape(self.images, [self.batchsize, self.imgdim*self.imgdim])
        generated_flattened = tf.reshape(self.G, [self.batchsize, self.imgdim*self.imgdim])
        vae_generation_loss = -tf.reduce_sum(flattened_images * tf.log(1e-10 + generated_flattened) + (1-flattened_images) * tf.log(1e-10 + 1 - generated_flattened),1)
        vae_latent_loss = 0.5 * tf.reduce_sum(tf.square(self.z_mean) + tf.square(self.z_stddev) - tf.log(tf.square(self.z_stddev)) - 1,1)
        self.vae_loss = tf.reduce_mean(vae_generation_loss + vae_latent_loss)



        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]
        vae_vars = [var for var in t_vars if 'vae_' in var.name]

        self.d_optim = tf.train.AdamOptimizer(self.learningrate_d, beta1=self.beta1).minimize(self.d_loss, var_list=d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.learningrate_g, beta1=self.beta1).minimize(self.g_loss, var_list=g_vars)
        self.vae_optim = tf.train.AdamOptimizer(self.learningrate_vae, beta1=self.beta1).minimize(self.vae_loss, var_list=vae_vars)

        init = tf.initialize_all_variables()
        # Launch the session
        self.sess = tf.Session()
        self.sess.run(init)

    def generator(self):
        z2 = dense(self.z, self.zdim, 7*7*self.generator_size*4, scope='g_h0_lin')
        h0 = tf.nn.relu(self.g_bn0(tf.reshape(z2, [-1, 7, 7, self.generator_size*4]))) # 4x4x256
        h1 = tf.nn.relu(self.g_bn1(conv_transpose(h0, [self.batchsize, 14, 14, self.generator_size*2], "g_h1"))) #8x8x128
        h4 = conv_transpose(h1, [self.batchsize, 28, 28, 1], "g_h4")
        return tf.nn.sigmoid(h4)

    def discriminator(self, image, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        # this is for MNIST
        # it starts at 28x28x1
        h0 = lrelu(conv2d(image, 1, self.discriminator_size, name='d_h0_conv')) #14x14x64
        h1 = lrelu(self.d_bn1(conv2d(h0, self.discriminator_size, self.discriminator_size*2, name='d_h1_conv'))) #7x7x128
        h4 = dense(tf.reshape(h1, [self.batchsize, -1]), 7*7*self.discriminator_size*2, 1, scope='d_h3_lin')
        return tf.nn.sigmoid(h4)

    def encoder(self,input_images):
        with tf.variable_scope("encoder"):
            w1 = tf.get_variable("vae_w1",[self.imgdim * self.imgdim,self.encoder_size])
            b1 = tf.get_variable("vae_b1",[self.encoder_size])
            w2 = tf.get_variable("vae_w2",[self.encoder_size,self.encoder_size])
            b2 = tf.get_variable("vae_b2",[self.encoder_size])
            w_mean = tf.get_variable("vae_w_mean",[self.encoder_size,self.zdim])
            b_mean = tf.get_variable("vae_b_mean",[self.zdim])
            w_stddev = tf.get_variable("vae_w_stddev",[self.encoder_size,self.zdim])
            b_stddev = tf.get_variable("vae_b_stddev",[self.zdim])

        flattened = tf.reshape(input_images,[self.batchsize,self.imgdim * self.imgdim])
        h1 = tf.nn.sigmoid(tf.matmul(flattened,w1) + b1)
        h2 = tf.nn.sigmoid(tf.matmul(h1,w2) + b2)
        o_mean = tf.matmul(h2,w_mean) + b_mean
        o_stddev = tf.matmul(h2,w_stddev) + b_stddev
        return o_mean, o_stddev

    def train(self):
        train_loss_d = train_loss_g = train_loss_vae = 0.5
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        n_samples = mnist.train.num_examples
        for epoch in range(10):
            for idx in range(int(n_samples / self.batchsize)):
                # z = np.random.uniform(-1.0,1.0, size=(self.batchsize, self.zdim)).astype(np.float32)
                batch = mnist.train.next_batch(self.batchsize)[0].reshape([self.batchsize,self.imgdim,self.imgdim,1])
                for i in xrange(4):
                    train_loss_g, _ = self.sess.run([self.g_loss, self.g_optim],feed_dict={self.images: batch})

                train_loss_d, _ = self.sess.run([self.d_loss, self.d_optim],feed_dict={self.images: batch})

                train_loss_vae, _ = self.sess.run([self.vae_loss, self.vae_optim],feed_dict={self.images: batch})

                print "%d: %f %f %f" % (idx, train_loss_d, train_loss_g, train_loss_vae)

                if idx % 15 == 0:
                    generated_test = self.sess.run(self.G, feed_dict={self.images: batch})
                    generated_test = generated_test.reshape(self.batchsize,28,28)
                    ims("results/"+str(idx + epoch*100000)+".jpg",merge(generated_test[:64],[8,8]))

model = Model()
model.train()
