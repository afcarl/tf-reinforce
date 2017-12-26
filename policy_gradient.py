""" TensorFlow implementation of policy gradient (REINFORCE). """

import argparse
import gym
import roboschool
import numpy as np
import tensorflow as tf
# quick fix for roboschool
from OpenGL import GL

parser = argparse.ArgumentParser(description="TensorFlow implementation of Policy Gradient")
parser.add_argument("--n_eps", type=int, default=1000, help="Number of episodes for training.")
parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor for rewards.")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for Adam optimizer.")
parser.add_argument("--env_id", type=str, default="RoboschoolHumanoid-v1",
                    help="OpenAI Roboschool Gym environment ID.")
args = parser.parse_args()

# Available OpenAI Gym environments from Roboschool.
# If the argument environment is not in this list, run with the default
# environment (RoboschoolHumanoid-v1).
roboschool_envs = [s.id for s in gym.envs.registry.all() if s.id.startswith("Roboschool")]
env_id = args.env_id if args.env_id in roboschool_envs else "RoboschoolHumanoid-v1"

# Initialize OpenAI Gym environment, and configure state size, hidden size, and the action size,
# according to the environment. Note that all environments from Roboschool has Box for observation
# spaces and action spaces.
env = gym.make(env_id)
obs_size = env.observation_space.shape[0]
act_size = env.action_space.shape[0]


def weight_variable(shape, name):
    """ Return a tensorflow.Variable for dense layer weights. """
    init_val = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_val, name=name)


def bias_variable(shape, name):
    """ Return a tensorflow.Variable for bias. """
    return tf.Variable(tf.constant(0.1, shape=shape), name=name)


class Policy(object):
    """ Policy network with two layers. """

    def __init__(self, obs_size, act_size):
        self.sess = tf.Session()

        # Observation represents state argument (s) of the policy.
        # In terms of a neural network, this is the input layer.
        with tf.name_scope("observation"):
            self.observation = tf.placeholder(tf.float32, (None, obs_size))

        # Hidden layer of the policy network.
        # This layer has the same dimensions as the input observation dimensions.
        with tf.name_scope("hidden"):
            W_1 = weight_variable((obs_size, obs_size), "W_1")
            b_1 = bias_variable((obs_size), "b_1")
            a_1 = tf.nn.relu(tf.matmul(self.observation, W_1) + b_1)

        # Output layer of the policy network.
        # This layer returns the probability of each action.
        with tf.name_scope("output"):
            W_2 = weight_variable((obs_size, act_size), "W_2")
            b_2 = bias_variable((act_size), "b_2")          
            a_2 = tf.matmul(a_1, W_2) + b_2

        with tf.name_scope("action"):
            # Index of the chosen action based on the its probability.
            self.action = tf.placeholder(tf.int32, (None, ))
            self.pi_sa = tf.gather(tf.softmax(a_2), self.action)

        with tf.name_scope("reward"):
            self.reward = tf.placeholder(tf.float32, (None, ))

        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(tf.log(pi_sa) * self.reward)

        with tf.name_scope("train_op"):
            optimizer = tf.train.AdamOptimizer(args.lr)
            self.train_op = optimizer.minimize(-self.loss)

    def __call__(self, state):
        """ Evaluate P{a = action | s = state} for all action. """
        return self.sess.run(self.pi_sa, feed_dict={self.observation: state})

    def optimize(self, state, reward):
        """ Optimize the policy given a set of states and corresponding rewards. """
        feed_dict = {self.observation: state, self.reward: reward}
        loss, _ = self.run([self.loss, self.train_op], feed_dict=feed_dict)
        return loss

class Trajectory(object):
    """ Sequence of (state, action, reward). """

    def __init__(self):
        self._state_buffer = []
        self._action_buffer = []
        self._reward_buffer = []
        self._len = 0

    def __str__(self):
        return "Trajectory(%d)" % self._len

    def __len__(self):
        return self._len

    def __getitem__(self, t):
        """ Return a tuple of (state, action, reward) at time t. """
        if type(t) is not int:
            raise TypeError("Trajectory indices must be integers, not %s" % type(t))
        if t not in range(0, self._len):
            raise IndexError("Trajectory index out of range")
        return (self._state_buffer[t], self._action_buffer[t], self._reward_buffer[t])

    def append(self, s, a, r):
        """ Append a set of state, action, and reward. """
        self._state_buffer.append(s)
        self._action_buffer.append(a)
        self._reward_buffer.append(r)
        self._len += 1


