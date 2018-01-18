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

    def __getitem__(self, t):
        if t > self._T - 1:
            raise IndexError("Episode index out of range")
        return (self._S[t], self._A[t], self._R[t])

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
            return np.stack(returns, axis=0) / norm
        return np.stack(returns, axis=0)

def weight_variable(shape, name):
    init_val = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_val, name=name)

def bias_variable(shape, name):
    return tf.Variable(tf.constant(0.1, shape=shape), name=name)

class Policy(object):
    def __init__(self, state_dim, act_dim):
        self.sess = tf.Session()
        with tf.name_scope("state"):
            self.S = tf.placeholder(tf.float32, [state_dim], name="S")
        with tf.name_scope("policy_network"):
            with tf.name_scope("layer_1"):
                W_1 = weight_variable([state_dim, state_dim], "W_1")
                b_1 = bias_variable([state_dim], "b_1")
                a_1 = tf.nn.relu(tf.matmul(tf.expand_dims(self.S, 0), W_1) + b_1)
            with tf.name_scope("layer_2"):
                W_2 = weight_variable([state_dim, act_dim], "W_2")
                b_2 = bias_variable([act_dim], "b_2")
                self.act_probs = tf.squeeze(tf.nn.softmax(tf.matmul(a_1, W_2) + b_2))
        with tf.name_scope("action"):
            self.A = tf.placeholder(tf.int32, name="A")
            self.pi_as = tf.gather(self.act_probs, self.A, name="pi_as")
        with tf.name_scope("objective"):
            self.G = tf.placeholder(tf.float32, name="G")
            self.objective = tf.log(self.pi_as) * self.G
            tf.summary.scalar("objective", self.objective)
        with tf.name_scope("train_op"):
            optimizer = tf.train.AdamOptimizer(0.01)
            self.train_op = optimizer.minimize(-self.objective)
        self.merged = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter("log", graph=self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def __call__(self, state):
        """ Evaluate the policy given the argument state, and take an action. """
        return self.sess.run(self.act_probs, feed_dict={self.S: state})

    def train_step(self, ep, S, A, G):
        """ Train the policy network given the argument states and returns. """
        feed_dict = {self.S: S, self.A: A, self.G: G}
        summary, _ = self.sess.run([self.merged, self.train_op], feed_dict=feed_dict)
        self.summary_writer.add_summary(summary, ep)

def rollout(policy, env, render=False):
    """ Return an episode of the argument policy's rollout in the argument environment. """
    episode = Episode()
    s = env.reset()
    done = False
    while not done:
        if render:
            env.render()
        act_probs = policy(s)
        a = np.random.choice(np.arange(len(act_probs)), p=act_probs)
        s_, r, done, info = env.step(a)
        episode.append(s, a, r)
        s = s_
    return episode

def reinforce(policy, env, gamma, n_eps):
    for i in trange(n_eps):
        episode = rollout(policy, env)
        returns = episode.returns(gamma)
        for t, (s, a, r) in enumerate(episode):
            policy.train_step(i, s, a, returns[t])

env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
act_dim = env.action_space.n
policy = Policy(state_dim, act_dim)
reinforce(policy, env, 0.99, 5000)
input("Training complete, press ENTER when ready.")
rollout(policy, env, render=True)