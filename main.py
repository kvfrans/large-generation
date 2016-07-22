import tensorflow as tf
import numpy as np
from ops import *
from scipy.misc import imsave as ims
import input_data

class Model():
    def __init__(self, batchsize=64, zdim=32, cdim = 1, scale=8):
        self.batchsize = batchsize
        self.zdim = zdim
        self.cdim = cdim
        self.scale = scale
        self.size = 128
        self.xdim = 26
        self.ydim = 26
        self.df_dim = 24
        self.beta1 = 0.65
        self.learningrate_d = 0.001
        self.learningrate_g = 0.001
        self.learningrate_vae = 0.001


        self.images = tf.placeholder(tf.float32,[self.batchsize,self.xdim,self.ydim,self.cdim])
        self.batch_flatten = tf.reshape(self.images, [self.batchsize, -1])
        self.npoints = self.xdim*self.ydim
        # self.xvec, self.yvec = self.coordinates(self.xdim,self.ydim,self.scale)



        # none = npoints
        self.x = tf.placeholder(tf.float32, [self.batchsize, None, 1])
        self.y = tf.placeholder(tf.float32, [self.batchsize, None, 1])
        self.r = tf.placeholder(tf.float32, [self.batchsize, None, 1])

        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')

        # self.g_bn0 = batch_norm(name='g_bn0')
        # self.g_bn1 = batch_norm(name='g_bn1')
        # self.g_bn2 = batch_norm(name='g_bn2')
        # self.g_bn3 = batch_norm(name='g_bn3')


        self.z_mean, self.z_log_sigma_sq = self.encoder()
        eps = tf.random_normal((self.batchsize, self.zdim), 0, 1, dtype=tf.float32)
        # z = mu + sigma*epsilon
        self.z = tf.add(self.z_mean, tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        # generate images from self.z, self.xdim, self.ydim
        # self.G = self.generator_absolute()
        self.G = self.generator()
        self.batch_reconstruct_flatten = tf.reshape(self.G, [self.batchsize, -1])

        # try to discriminate real images as 1
        D_prob = self.discriminator(self.images, reuse=False)
        # try to discriminate fake images as 0
        D_fake_prob = self.discriminator(self.G, reuse=True)

        self.create_vae_loss_terms()

        self.d_loss_real = binary_cross_entropy_with_logits(tf.ones_like(D_prob), D_prob)
        self.d_loss_fake = binary_cross_entropy_with_logits(tf.zeros_like(D_fake_prob), D_fake_prob)
        self.dloss = 1.0*(self.d_loss_real + self.d_loss_fake)/ 2.0
        self.gloss = 1.0*binary_cross_entropy_with_logits(tf.ones_like(D_fake_prob), D_fake_prob)

        # d_loss_real = tf.reduce_mean(binary_cross_entropy_with_logits(tf.ones_like(D_prob),D_prob))
        # d_loss_fake = tf.reduce_mean(binary_cross_entropy_with_logits(tf.zeros_like(D_fake_prob),D_fake_prob))
        #
        # self.gloss = tf.reduce_mean(binary_cross_entropy_with_logits(tf.ones_like(D_fake_prob),D_fake_prob))
        # self.dloss = 0.5*d_loss_real + 0.5*d_loss_fake

        self.balanced_loss = 1.0 * self.gloss + 1.0 * self.vae_loss # can try to weight these.

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]
        q_vars = [var for var in t_vars if 'q_' in var.name]
        vae_vars = q_vars+g_vars
        print [var.name for var in t_vars if 'g_' in var.name]

        self.d_optim = tf.train.AdamOptimizer(self.learningrate_d, beta1=self.beta1).minimize(self.dloss, var_list=d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.learningrate_g, beta1=self.beta1).minimize(self.gloss, var_list=g_vars)
        self.vae_optim = tf.train.AdamOptimizer(self.learningrate_vae, beta1=self.beta1).minimize(self.vae_loss, var_list=vae_vars)

        init = tf.initialize_all_variables()
        # Launch the session
        self.sess = tf.Session()
        self.sess.run(init)

    def create_vae_loss_terms(self):
        # The loss is composed of two terms:
        # 1.) The reconstruction loss (the negative log probability
        #     of the input under the reconstructed Bernoulli distribution
        #     induced by the decoder in the data space).
        #     This can be interpreted as the number of "nats" required
        #     for reconstructing the input when the activation in latent
        #     is given.
        # Adding 1e-10 to avoid evaluatio of log(0.0)
        self.reconstr_loss = \
            -tf.reduce_sum(self.batch_flatten * tf.log(1e-10 + self.batch_reconstruct_flatten)
                           + (1-self.batch_flatten) * tf.log(1e-10 + 1 - self.batch_reconstruct_flatten), 1)


        self.nantest = 1e-10 + 1 - self.batch_reconstruct_flatten
        self.nantest2 = tf.log(1e-10 + 1 - self.batch_reconstruct_flatten)

        # 2.) The latent loss, which is defined as the Kullback Leibler divergence
        ##    between the distribution in latent space induced by the encoder on
        #     the data and some prior. This acts as a kind of regularizer.
        #     This can be interpreted as the number of "nats" required
        #     for transmitting the the latent space distribution given
        #     the prior.
        self.latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq
                                           - tf.square(self.z_mean)
                                           - tf.exp(self.z_log_sigma_sq), 1)
        self.vae_loss = tf.reduce_mean(self.reconstr_loss + self.latent_loss) / self.npoints # average over batch and pixel

    def encoder(self):
        # Generate probabilistic encoder (recognition network), which
        # maps inputs onto a normal distribution in latent space.
        # The transformation is parametrized and can be learned.
        H1 = tf.nn.dropout(tf.nn.softplus(linear(self.batch_flatten, 256, 'q_lin1')), 1.0)
        H2 = tf.nn.dropout(tf.nn.softplus(linear(H1, 256, 'q_lin2')), 1.0)
        z_mean = linear(H2, self.zdim, 'q_lin3_mean')
        z_log_sigma_sq = linear(H2, self.zdim,'q_lin3_log_sigma_sq')
        return (z_mean, z_log_sigma_sq)

    def coordinates(self, xdim, ydim, scale):
        n_points = xdim * ydim
        x_range = scale*(np.arange(xdim)-(xdim-1)/2.0)/(xdim-1)/0.5
        y_range = scale*(np.arange(ydim)-(ydim-1)/2.0)/(ydim-1)/0.5
        x_mat = np.matmul(np.ones((ydim, 1)), x_range.reshape((1, xdim)))
        y_mat = np.matmul(y_range.reshape((ydim, 1)), np.ones((1, xdim)))
        r_mat = np.sqrt(x_mat*x_mat + y_mat*y_mat)
        x_mat = np.tile(x_mat.flatten(), self.batchsize).reshape(self.batchsize, n_points, 1)
        y_mat = np.tile(y_mat.flatten(), self.batchsize).reshape(self.batchsize, n_points, 1)
        r_mat = np.tile(r_mat.flatten(), self.batchsize).reshape(self.batchsize, n_points, 1)
        return x_mat, y_mat, r_mat

    def generator(self, reuse = False):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        z_scaled = tf.reshape(self.z, [self.batchsize, 1, self.zdim]) * tf.ones([self.npoints, 1], dtype=tf.float32) * self.scale
        z_unroll = tf.reshape(z_scaled, [self.batchsize*self.npoints, self.zdim])
        x_unroll = tf.reshape(self.x, [self.batchsize*self.npoints, 1])
        y_unroll = tf.reshape(self.y, [self.batchsize*self.npoints, 1])
        r_unroll = tf.reshape(self.r, [self.batchsize*self.npoints, 1])

        U = fully_connected(z_unroll, self.size, 'g_z') + \
        fully_connected(x_unroll, self.size, 'g_x', with_bias = False) + \
        fully_connected(y_unroll, self.size, 'g_y', with_bias = False) + \
        fully_connected(r_unroll, self.size, 'g_r', with_bias = False)

        H = tf.nn.softplus(U)
        h2 = tf.nn.tanh(fully_connected(H, self.size, 'g_2'))
        h3 = tf.nn.tanh(fully_connected(h2, self.size, 'g_3'))
        h4 = tf.nn.tanh(fully_connected(h3, self.size, 'g_4'))
        h5 = tf.nn.tanh(fully_connected(h4, self.size, 'g_5'))
        output = fully_connected(h5, self.cdim, 'g_final')
        result = tf.reshape(output, [self.batchsize, self.ydim, self.xdim, self.cdim])
        return tf.nn.sigmoid(result)

    def generator_absolute(self):
        z2 = dense(self.z, self.zdim, 7*7*self.df_dim*4, scope='g_h0_lin')
        h0 = tf.nn.relu(self.g_bn0(tf.reshape(z2, [-1, 7, 7, self.df_dim*4]))) # 4x4x256
        h1 = tf.nn.relu(self.g_bn1(conv_transpose(h0, [self.batchsize, 14, 14, self.df_dim*2], "g_h1"))) #8x8x128
        h4 = conv_transpose(h1, [self.batchsize, 28, 28, 1], "g_h4")
        return tf.nn.sigmoid(h4)

    def generate(self, z=None):
        if z is None:
            z = np.random.normal(-1.0,1.0, size=(self.batchsize, self.zdim)).astype(np.float32)

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

    def train(self,batch):
        x_vec, y_vec, r_vec = self.coordinates(self.xdim, self.ydim, self.scale)

        closs_d = closs_g = closs_vae = 0.5
        # print batch[0,:,:,0]

        for i in xrange(4):
            closs_g, _ = self.sess.run([self.gloss, self.g_optim],feed_dict={ self.x: x_vec, self.y: y_vec, self.r: r_vec, self.images: batch})

        # print self.sess.run(self.vae_loss,feed_dict={ self.x: x_vec, self.y: y_vec, self.r: r_vec, self.images: batch})
        # print self.sess.run(self.reconstr_loss,feed_dict={ self.x: x_vec, self.y: y_vec, self.r: r_vec, self.images: batch})
        # print self.sess.run(self.latent_loss,feed_dict={ self.x: x_vec, self.y: y_vec, self.r: r_vec, self.images: batch})
        print self.sess.run(self.batch_reconstruct_flatten,feed_dict={ self.x: x_vec, self.y: y_vec, self.r: r_vec, self.images: batch})
        print self.sess.run(self.batch_flatten,feed_dict={ self.x: x_vec, self.y: y_vec, self.r: r_vec, self.images: batch})
        print "nantest"
        prenan = self.sess.run(self.nantest,feed_dict={ self.x: x_vec, self.y: y_vec, self.r: r_vec, self.images: batch})
        print prenan
        print "nantest2"
        print np.log(prenan)
        print "nan3"
        print self.sess.run(self.nantest2,feed_dict={ self.x: x_vec, self.y: y_vec, self.r: r_vec, self.images: batch})




        # for i in xrange(4):
            # closs_vae, _ = self.sess.run([self.vae_loss, self.vae_optim],feed_dict={ self.x: x_vec, self.y: y_vec, self.r: r_vec, self.images: batch})
        if closs_g < 0.75:
            closs_d, _ = self.sess.run([self.dloss, self.d_optim],feed_dict={ self.x: x_vec, self.y: y_vec, self.r: r_vec, self.images: batch})


        # if idx % 50 == 0:
        #     print self.generate()[0,:,:,0]
        #     ims("results/"+str(idx)+".jpg",self.generate()[0,:,:,0])
        #     ims("results/o.jpg",troll)
        return closs_d, closs_g, closs_vae



# model = Model()
# model.train()
