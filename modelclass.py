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

  

    def neural_net(self, X, weights, biases):

        H = H = 2.0*(X-self.lb)/(self.ub-self.lb)-1.0#X

        for l in range(len(weights)-1):

            W = weights[l]
            b = biases[l]

            H = tf.nn.relu(tf.add(tf.matmul(H,W),b))#tf.tanh(tf.add(tf.matmul(H, W), b))

        W = weights[-1]
        b = biases[-1]

        Y = tf.add(tf.matmul(H, W), b)

        return Y

    def net_projectile(self, t, weights, biases):

        u = self.neural_net(t, weights, biases)

        x = u[:,0:1]
        y = u[:,1:2]

        x_t = tf.gradients(x, t)[0]
        y_t = tf.gradients(y, t)[0]

        x_tt = tf.gradients(x_t, t)[0]
        y_tt = tf.gradients(y_t, t)[0]

        f_x = x_tt
        f_y = y_tt + self.g

        f = tf.concat([f_x, f_y], axis=1)

        return u, f

    def train(self, nIter):

        tf_dict = {
            self.t_tf: self.t,
            self.u_tf: self.u
        }

        for it in range(nIter):

            self.sess.run(self.train_op_Adam, tf_dict)

            if it % 500 == 0:

                loss_value = self.sess.run(self.loss, tf_dict)

                print("Iteration:", it,
                      "Loss:", loss_value)

    def predict(self, t_star):

        tf_dict = {
            self.t_tf: t_star
        }

        u_star = self.sess.run(self.u_pred, tf_dict)

        return u_star
