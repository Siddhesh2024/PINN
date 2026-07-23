import sys
sys.path.insert(0, '../../Utilities/')

import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
import time

tf.disable_v2_behavior()

np.random.seed(1234)
tf.set_random_seed(1234)
nu=0.01/np.pi

class PhysicsInformedNN:
    def __init__(self, x, t, u, x_f, t_f, layers):

        X = np.concatenate([x, t], 1)
        X_f = np.concatenate([x_f, t_f], 1)

        X_all = np.concatenate([X, X_f], 0)

        self.lb = X_all.min(0)
        self.ub = X_all.max(0)

        self.X = X

        self.x = X[:,0:1]
        self.t = X[:,1:2]

        self.x_x_f= x_f
        self.t_t_f= t_f

        self.u = u

        self.layers = layers

        # Initialize NN
        self.weights, self.biases = self.initialize_NN(layers)

        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])

        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])

        self.x_x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_x_f.shape[1]])
        self.t_t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_t_f.shape[1]])

        self.u_pred, self.f_u_pred= self.net_NS(self.x_tf, self.t_tf, self.x_x_f_tf, self.t_t_f_tf)

        self.loss = tf.reduce_sum(tf.square(self.u_tf - self.u_pred)) + \
                    tf.reduce_sum(tf.square(self.f_u_pred))

        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1

        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_NS(self, x, t, x_f, t_f):

        u = self.neural_net(tf.concat([x,t], 1), self.weights, self.biases)

        u_f = self.neural_net(tf.concat([x_f,t_f], 1), self.weights, self.biases)


        u_t = tf.gradients(u_f, t_f)[0]
        u_x = tf.gradients(u_f, x_f)[0]
        u_xx = tf.gradients(u_x, x_f)[0]

        f_u = u_t + (u_f*u_x) - nu*(u_xx)

        return u, f_u

    def callback(self, loss):
        print('Loss: %.3e' % (loss))

    def train(self, nIter):

        tf_dict = {self.x_tf: self.x, self.t_tf: self.t,
                   self.u_tf: self.u,
                   self.x_x_f_tf: self.x_x_f, self.t_t_f_tf: self.t_t_f}

        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)

            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e' %
                      (it, loss_value))
                start_time = time.time()

        #self.optimizer.minimize(self.sess,
         #                       feed_dict = tf_dict,
          #                      fetches = [self.loss],#, self.lambda_1, self.lambda_2],
           #                     loss_callback = self.callback)


    def predict(self, x_star, t_star):

        tf_dict = {self.x_tf: x_star, self.t_tf: t_star}

        u_star = self.sess.run(self.u_pred, tf_dict)

        return u_star
