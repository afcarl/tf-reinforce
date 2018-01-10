""" TensorFlow implementation of REINFORCE. """

import os
# Silence TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
from tqdm import trange
import numpy as np
import tensorflow as tf
import gym

class Episode(object):
    def __init__(self):
        self._S = []
        self._A = []
        self._R = []
        self._T = 0

    def __len__(self):
        return self._T

    @property
    def states(self):
        return self._S

    @property
    def actions(self):
        return self._A

    @property
    def rewards(self):
        return self._R

    def append(self, s, a, r):
        self._S.append(s)
        self._A.append(a)
        self._R.append(r)
        self._T += 1 

    def returns(self, gamma=0.99):
        """ Return a list of return values (G_t) at each time step t. """
        g_t = 0
        returns = np.zeros_like(self._R)
        for t in reversed(range(self._T)):
            g_t = g_t * gamma + self._R[t]
            returns[t] = g_t
        # Normalize the returns
        norm = np.linalg.norm(returns)
        if norm != 0:
            return np.stack(returns, axis=0)[:, np.newaxis] / norm
        return np.stack(returns, axis=0)[:, np.newaxis]

class Policy(object):
    def __init__(self, state_dim, act_dim):
        with tf.name_scope("policy_network"):
            self.S = tf.placeholder(tf.float32, [None, state_dim], name="S")
            W = tf.Variable(tf.truncated_normal(shape=[state_dim, act_dim], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[act_dim]), "b")
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
        return self.sess.run(self.pi_as, feed_dict={self.S: state})

    def train(self, alpha, states, returns):
        """ Train the policy network given the argument states and returns. """
        self.sess.run(self.update, feed_dict={self.alpha: alpha, self.S: states, self.G: returns})

def rollout(policy, env, render=False):
    """ Return an episode of the argument policy's rollout in the argument environment. """
    episode = Episode()
    s = env.reset()
    done = False
    while not done:
        if render:
            env.render()
        a = policy(s[np.newaxis, :]).flatten()
        s_, r, done, info = env.step(a)
        episode.append(s, a, r)
        s = s_
    return episode

def reinforce(policy, env, n_eps):

    for i in trange(n_eps):
        ep = Episode()

env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
act_dim = env.action_space.n
policy = Policy(state_dim, act_dim)

for i in trange(1000):
    ep = Episode()
    s = env.reset()
    done = False
    while not done:
        env.render()
        act_prob = policy(s[np.newaxis, :])
        s_, r, done, _ = env.step(a)
        ep.append(s, a, r)
        s = s_
    
