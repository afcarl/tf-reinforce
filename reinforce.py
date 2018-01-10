""" TensorFlow implementation of REINFORCE. """

import numpy as np
import tensorflow as tf

class Episode(object):
    def __init__():
        self._S = []
        self._A = []
        self._R = []
        self._len = 0

    def __len__(self):
        return self._len

    def append(self, s, a, r):
        self._S.append(s)
        self._A.append(a)
        self._R.append(r)
        self._len += 1 

    def returns(self):
        """ Return a list of return values (G_t) at each time step t. """
        return

class Policy(object):
    def __init__(self, env_id):
        with tf.name_scope("policy_network"):
            self.S = tf.placeholder(tf.float32, [None, obs_dim], name="S")
            W = tf.Variable(tf.truncated_normal([obs_dim, act_dim], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, [act_dim]), "b")
            self.pi_as = tf.nn.softmax(tf.matmul(self.S, W) + b)
        with tf.name_scope("objective"):
            self.G = tf.placeholder(tf.float32, [None], name="G")
            self.objective = tf.reduce_mean(tf.log(self.pi_as) * self.G)
        with tf.name_scope("policy_gradient"):
            theta = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="policy_network")
            delta_theta = tf.gradients(self.objective, theta)
        with tf.name_scope("update"):
            self.alpha = tf.placeholder(tf.float32, name="alpha")
            self.update = [th.assign(th + self.alpha * dth) for th, dth in zip(theta, delta_theta)]
        self.sess = tf.Session()
        self.summary_writer = tf.summary.FileWriter("log", graph=self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def __call__(self, state):
        """ Evaluate the policy given the argument state, and take an action. """
        act_probs = tf.squeeze(self.pi_as)


    def update(self, alpha, states, returns):
        """ Train the policy network given the argument states and returns. """
        self.sess.run(self.update, feed_dict={self.alpha: alpha, self.S: states, self.G: returns})

env = gym.make("CartPole-v1")
for i in range(1000):
    ep = Episode()
    s = env.reset()
    done = False
    while not done:



