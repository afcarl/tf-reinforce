# TensorFlow implementation of Policy Gradient (REINFORCE)

import argparse
import gym
import roboschool
import numpy as np
import tensorflow as tf

from OpenGL import GL

parser = argparse.ArgumentParser(description="TensorFlow implementation of Policy Gradient")
parser.add_argument("--n_eps", type=int, default=1000, help="Number of episodes for training.")
parser.add_argument("--n_iter", type=int, default=10000, help="Number of iterations per episode.")
parser.add_argument("--env_id", type=str, default="RoboschoolHumanoid-v1",
                    help="OpenAI Roboschool Gym environment ID.")
args = parser.parse_args()

# Available OpenAI Gym environments from Roboschool.
# If the argument environment is not in this list, run with the default
# environment (RoboschoolHumanoid-v1).
roboschool_envs = [s.id for s in gym.envs.registry.all() if s.id.startswith("Roboschool")]
env_id = args.env_id if args.env_id in roboschool_envs else "RoboschoolHumanoid-v1"

# Initialize OpenAI Gym environment, and configure state size,
# hidden size, and the action size, according to the environment.
#
# Note that all environments from Roboschool has Box for observation
# spaces and action spaces.
env = gym.make(env_id)
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]

# A two-layered policy network with ReLU nonlinearity and a residual
# connection from the input to output layer, i.e., the sum of input and
# the hidden layer is forwarded to the output layer.
with tf.name_scope("policy"):
    # Observation of the state.
    with tf.name_scope("observation"):
        observation = tf.placeholder(tf.float32, (None, state_size))

    # Hidden layer of the policy network.
    with tf.name_scope("hidden_layer"):
        W_1 = tf.Variable(tf.truncated_normal([state_size, state_size], stddev=0.1), name="W_1")
        b_1 = tf.Variable(tf.constant(0.1, shape=[state_size]), name="b_1")
        hidden = tf.nn.relu(tf.matmul(observation, W_1) + b_1)

    # Output layer of the policy network.
    # The output of this network is not the action quite yet. If this policy is stochastic, the
    # output vector is used as the mean of action distribution (Gaussian).
    with tf.name_scope("output_layer"):
        W_2 = tf.Variable(tf.truncated_normal([state_size, action_size], stddev=0.1), name="W_2")
        b_2 = tf.Variable(tf.constant(0.1, shape=[action_size]), name="b_2")
        output = tf.matmul(hidden, W_2) + b_2

for _ in range(args.n_eps):
    state = env.reset()
    for t in range(args.n_iter):
        env.render()

        with tf.Session() as sess:
            action_mean = sess.run([output], feed_dict={observation: state})
            action = np.random.normal(loc=action_mean, scale=0.1)

        state, reward, done, info = env.step(action)

        # TODO: implement stochastic gradient ascend with policy gradient.

        if done:
            break
