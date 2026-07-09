import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt

tf.disable_v2_behavior()

class PINN:

    def __init__(self, t, u, layers):

        X = np.concatenate([t], 1)

        self.t = X[:,0:1]
        self.u = u

        self.weights, self.biases = self.initialize_NN(layers)

        self.g = 9.81

        self.sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False))

        self.t_tf = tf.placeholder(tf.float32,
                                   shape=[None, self.t.shape[1]])

        self.u_tf = tf.placeholder(tf.float32,
                                   shape=[None, self.u.shape[1]])

        self.u_pred, self.f_pred = self.net_projectile(
            self.t_tf,
            self.weights,
            self.biases)

        self.loss = tf.reduce_mean(tf.square(self.u_tf-self.u_pred))# + \
                   # tf.reduce_mean(tf.square(self.f_pred))

        self.optimizer = tf.train.AdamOptimizer()

        self.train_op_Adam = self.optimizer.minimize(self.loss)

        grads = tf.gradients(self.loss, self.weights)

        for g in grads:
           print(g)

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):

        weights = []
        biases = []

        for l in range(len(layers)-1):

            W = self.xavier_init([layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1, layers[l+1]], dtype=tf.float32))

            weights.append(W)
            biases.append(b)

        return weights, biases

    def xavier_init(self, size):

        in_dim = size[0]
        out_dim = size[1]

        xavier_stddev = np.sqrt(2/(in_dim+out_dim))

        return tf.Variable(
            tf.truncated_normal([in_dim, out_dim],
            stddev=xavier_stddev),
            dtype=tf.float32)

  

        return u_star
