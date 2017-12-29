""" TensorFlow implementation of policy gradient (REINFORCE). """

import argparse
import gym
import roboschool
import numpy as np
import tensorflow as tf
# quick fix for roboschool
from OpenGL import GL

def weight_variable(shape, name):
    """ Return a tensorflow.Variable for dense layer weights. """
    init_val = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_val, name=name)

def bias_variable(shape, name):
    """ Return a tensorflow.Variable for bias. """
    return tf.Variable(tf.constant(0.1, shape=shape), name=name)

class GaussianPolicy(object):
    """ Gaussian policy network with three layers. """

    def __init__(self, arch, sigma_sq):
        if len(arch) != 4:
            raise ValueError("Policy network architecture must contain 4 ints")

        # Observation represents state argument (s) of the policy (pi).
        # In terms of a neural network, this is the input layer of the policy network.
        with tf.name_scope("observation"):
            self.observation = tf.placeholder(tf.float32, (None, arch[0]))

        # First hidden layer of the policy network.
        with tf.name_scope("hidden_1"):
            W_1 = weight_variable((arch[0], arch[1]), "W_1")
            b_1 = bias_variable((arch[1], ), "b_1")
            a_1 = tf.nn.relu(tf.matmul(observation, W_1) + b_1)

        # Second hidden layer of the policy network.
        with tf.name_scope("hidden_2"):
            W_2 = weight_variable((arch[1], arch[2]), "W_2")
            b_2 = bias_variable((arch[2], ), "b_2")
            a_2 = tf.nn.relu(tf.matmul(a_1, W_2) + b_2)

        # Output layer of the policy network.
        # This layer returns the mean vector of the action distribution.
        with tf.name_scope("action_mean"):
            W_3 = weight_variable((arch[2], arch[3]), "W_3")
            b_3 = bias_variable((arch[3], ), "b_3")
            mu = tf.matmul(a_2, W_3) + b_3

        # Use the output of the policy network as the mean of the action disctribution.
        # The action is sampled from the distribution to compute gradient.
        with tf.name_scope("action"):
            self.pi_sa = tf.distributions.Normal(mu, sigma_sq)
            self.action = self.pi_sa.sample()

        # Reward placeholder for updating the policy during training.
        with tf.name_scope("reward"):
            self.reward = tf.placeholder(tf.float32, (None, ))

        # Loss function of Gaussian policy.
        # L = log(pi(a | s)) * V(s, a)
        with tf.name_scope("loss"):
            log_pi_sa = self.pi_sa.log_prob(self.action)
            self.loss = tf.reduce_mean(log_pi_sa * self.reward)

        # Training operator with Adam optimizer.
        # Note that the optimizer minimizes the negative loss, i.e., maximizes the loss.
        with tf.name_scope("train_op"):
            optimizer = tf.train.AdamOptimizer(args.lr)
            self.train_op = optimizer.minimize(-self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def __call__(self, state):
        """ Sample from the action distribution given the argument state. """
        return self.sess.run(self.action, feed_dict={self.observation: state})

    def learn(self, state, reward):
        """ Optimize the policy given a set of states and corresponding rewards. """
        feed_dict = {self.observation: state, self.reward: reward}
        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict=feed_dict)
        return loss

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
        return np.stack(self._reward_buffer, axis=0)

def main(args):
    """ Execute REINFORCE (Monte Carlo policy gradient). """

    # Available OpenAI Gym environments from Roboschool.
    # If the argument environment is not in this list, run with the default
    # environment (RoboschoolHumanoid-v1).
    roboschool_envs = [s.id for s in gym.envs.registry.all() if s.id.startswith("Roboschool")]
    env_id = args.env_id if args.env_id in roboschool_envs else "RoboschoolAnt-v1"
    
    # Initialize OpenAI Gym environment, and configure the observation size and the action size,
    # according to the environment. Note that all environments from Roboschool uses Box for 
    # observation spaces and action spaces.
    env = gym.make(env_id)
    arch = [
        env.observation_space.shape[0],
        args.hidden[0],
        args.hidden[1],
        env.action_space.shape[0]
    ]
    
    pi = GaussianPolicy(arch, args.var)
    for ep in range(args.n_eps):
        episode = Episode()
        state = env.reset()
        while True:
            action = pi(state[np.newaxis, :]).reshape((arch[3]))
            state_, reward, done, info = env.step(action)
            episode.append(state, action, reward)
            state = state_
            if done:
                states = trajectory.states
                rewards = trajectory.rewards
                pi.optimize(states, rewards)
                # Only print the training process 10 times.
                if ep % (args.n_eps / 10) == 0:
                    ret = tf.reduce_sum(rewards).eval(pi.sess)
                    print("Episode %d: return = %f" % (ep, ret))
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TensorFlow implementation of Policy Gradient")
    parser.add_argument("--hidden", type=int, nargs=2, help="Dimensions of 2 hidden layers.")
    parser.add_argument("--n_eps", type=int, default=1000, help="Number of episodes for training.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for rewards.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for optimizer.")
    parser.add_argument("--var", type=float, default=0.01, help="Variance for Gaussian policy.")
    parser.add_argument("--env_id", type=str, default="RoboschoolAnt-v1", 
                        help="OpenAI Roboschool Gym environment ID.")
    args = parser.parse_args()
    
    main(args)
