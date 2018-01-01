""" TensorFlow implementation of REINFORCE. """

import os
# Silence TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import tqdm
import argparse
import gym
import roboschool
import numpy as np
import tensorflow as tf
# Quick fix for roboschool
from OpenGL import GL

def weight_variable(shape):
    """ Return a tensorflow.Variable for dense layer weights. """
    init_val = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_val)

def bias_variable(shape):
    """ Return a tensorflow.Variable for bias. """
    return tf.Variable(tf.constant(0.1, shape=shape))

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)
        tf.summary.scalar("mean", mean)
        with tf.name_scope("stddev"):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar("stddev", stddev)
            tf.summary.scalar("max", tf.reduce_max(var))
            tf.summary.scalar("min", tf.reduce_min(var))
            tf.summary.histogram("histogram", var)

class GaussianPolicy(object):
    """ Gaussian policy network with three layers. """

    def __init__(self, arch):
        if len(arch) != 4:
            raise ValueError("Policy network architecture must contain 4 integers")

        # Observation represents state argument (s) of the policy (pi).
        # In terms of a neural network, this is the input layer of the policy network.
        with tf.name_scope("observation"):
            self.observation = tf.placeholder(tf.float32, [None, arch[0]])

        # First hidden layer of the policy network.
        with tf.name_scope("hidden_1"):
            with tf.name_scope("W_1"):
                W_1 = weight_variable([arch[0], arch[1]])
                variable_summaries(W_1)
            with tf.name_scope("b_1"):
                b_1 = bias_variable([arch[1]])
                variable_summaries(b_1)
            a_1 = tf.nn.relu(tf.matmul(self.observation, W_1) + b_1)

        # Second hidden layer of the policy network.
        with tf.name_scope("hidden_2"):
            with tf.name_scope("W_2"):
                W_2 = weight_variable([arch[1], arch[2]])
                variable_summaries(W_2)
            with tf.name_scope("b_2"):
                b_2 = bias_variable([arch[2]])
                variable_summaries(b_2)
            a_2 = tf.nn.relu(tf.matmul(a_1, W_2) + b_2)

        # Output layer for the mean of the action distribution.
        with tf.name_scope("action_mu"):
            with tf.name_scope("W_3"):
                W_3 = weight_variable([arch[2], arch[3]])
                variable_summaries(W_3)
            with tf.name_scope("b_3"):
                b_3 = bias_variable([arch[3]])
                variable_summaries(b_3)
            with tf.name_scope("mu"):
                self.mu = tf.matmul(a_2, W_3) + b_3
                tf.summary.histogram("mu", self.mu)

        ## Output layer for the variance of the action distribution.
        #with tf.name_scope("action_sigma_sq"):
        #    with tf.name_scope("W_4"):
        #        W_4 = weight_variable([arch[2], arch[3]])
        #        variable_summaries(W_4)
        #    with tf.name_scope("b_4"):
        #        b_4 = bias_variable([arch[3]])
        #        variable_summaries(b_4)
        #    with tf.name_scope("sigma_sq"):
        #        self.sigma_sq = tf.matmul(a_2, W_4) + b_4
        #        tf.summary.histogram("sigma_sq", self.sigma_sq)

        # Use the output of the policy network as the mean of the action disctribution.
        # The action is sampled from the distribution to compute gradient.
        with tf.name_scope("action"):
            self.pi_sa = tf.distributions.Normal(self.mu, self.sigma_sq)
            self.action = self.pi_sa.sample()
            tf.summary.histogram("action", self.action)

        # Return value G_t at each time step t for computing loss.
        with tf.name_scope("G_t"):
            self.G_t = tf.placeholder(tf.float32, [None, 1])

        # Loss function of Gaussian policy.
        # L = log(pi(a | s)) * V(s, a)
        with tf.name_scope("loss"):
            log_pi_sa = self.pi_sa.log_prob(self.action)
            self.loss = -tf.reduce_mean(log_pi_sa * self.G_t)
            tf.summary.scalar("loss", self.loss)

        # Training operator with Adam optimizer.
        # Note that the optimizer minimizes the negative loss, i.e., maximizes the loss.
        with tf.name_scope("train_op"):
            optimizer = tf.train.AdamOptimizer(args.lr)
            self.train_op = optimizer.minimize(self.loss)

        self.merged = tf.summary.merge_all()
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        self.summary_writer = tf.summary.FileWriter("./logs", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def __call__(self, state):
        """ Sample from the action distribution given the argument state. """
        return self.sess.run(self.action, feed_dict={self.observation: state})

    def learn(self, episode, state, returns):
        """ Optimize the policy given a set of states and corresponding rewards. """
        feed_dict = {self.observation: state, self.G_t: returns}
        summary, _ = self.sess.run([self.merged, self.train_op], feed_dict=feed_dict)
        self.summary_writer.add_summary(summary, episode)

class Episode(object):
    """ Sequence of (state, action, reward). """

    def __init__(self):
        self._state_buffer = []
        self._action_buffer = []
        self._reward_buffer = []
        self._len = 0

    def __str__(self):
        return "Episode(%d)" % self._len

    def __len__(self):
        return self._len

    def __getitem__(self, t):
        """ Return a tuple of (state, action, reward) at time t. """
        if type(t) is not int:
            raise TypeError("Episode indices must be integers, not %s" % type(t))
        if t not in range(0, self._len):
            raise IndexError("Episode index out of range")
        return (self._state_buffer[t], self._action_buffer[t], self._reward_buffer[t])

    def append(self, s, a, r):
        """ Append a set of state, action, and reward. """
        self._state_buffer.append(s)
        self._action_buffer.append(a)
        self._reward_buffer.append(r)
        self._len += 1

    @property
    def states(self):
        """ Return the state tensor. """
        return np.stack(self._state_buffer, axis=0)

    @property
    def actions(self):
        """ Return the action tensor. """
        return np.stack(self._action_buffer, axis=0)

    @property
    def rewards(self):
        """ Return the reward vector. """
        return np.stack(self._reward_buffer, axis=0)[:, np.newaxis]

    def returns(self, gamma):
        """ Return a vector of return values at each time step. """
        g_t = 0
        returns = np.zeros_like(self._reward_buffer)
        for t in reversed(range(self._len)):
            g_t = g_t * gamma + self._reward_buffer[t]
            returns[t] = g_t
        # Normalize the returns
        norm = np.linalg.norm(returns)
        if norm != 0:
            return np.stack(returns, axis=0)[:, np.newaxis] / norm
        return np.stack(returns, axis=0)[:, np.newaxis]
        

def reinforce(pi, env, n_eps, gamma):
    """ Execute REINFORCE (Monte Carlo policy gradient) to train the argument policy (pi) in
    the argument environment. """
    for ep in tqdm.trange(n_eps):
        episode = rollout(pi, env)
        pi.learn(ep, episode.states, episode.returns(gamma))

def rollout(pi, env, render=False):
    """ Make the argument policy interact with the environment. """
    episode = Episode()
    state = env.reset()
    done = False
    while not done:
        if render:
            env.render()
        action = pi(state[np.newaxis, :]).flatten()
        state_, reward, done, info = env.step(action)
        episode.append(state, action, reward)
        state = state_
    return episode

def export_model():
    """ Export the trained model. """

    # TODO: implementation

    return

def main(args):
    # Available OpenAI Gym environments from Roboschool.
    # If the argument environment is not in this list, run with the default
    # environment (RoboschoolHumanoid-v1).
    roboschool_envs = [s.id for s in gym.envs.registry.all() if s.id.startswith("Roboschool")]
    env_id = args.env_id if args.env_id in roboschool_envs else "RoboschoolAnt-v1"
    
    # Initialize OpenAI Gym environment, and configure the observation size and the action size,
    # according to the environment. Note that all environments from Roboschool uses Box for 
    # observation spaces and action spaces.
    env = gym.make(env_id)
    arch = [env.observation_space.shape[0],
            args.hidden[0],
            args.hidden[1],
            env.action_space.shape[0]]

    # Print summary of the experiment setup.
    print("Environment ID:", env_id)
    print("Policy architecture:", arch)

    # Run REINFORCE.
    pi = GaussianPolicy(arch)
    reinforce(pi, env, args.n_eps, args.gamma)

    # Visually test the trained policy.
    input("Training complete, press ENTER when ready.")
    rollout(pi, env, render=True)    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TensorFlow implementation of Policy Gradient")
    parser.add_argument("--hidden", type=int, nargs=2, help="Dimensions of 2 hidden layers.")
    parser.add_argument("--n_eps", type=int, default=1000, help="Number of episodes for training.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for rewards.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for optimizer.")
    parser.add_argument("--env_id", type=str, default="RoboschoolAnt-v1", 
                        help="OpenAI Roboschool Gym environment ID.")
    args = parser.parse_args()
    
    main(args)
