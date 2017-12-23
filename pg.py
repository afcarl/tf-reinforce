# TensorFlow implementation of policy gradient

import argparse
import gym
import roboschool
import numpy as np
import tensorflow as tf
# quick fix for roboschool
from OpenGL import GL

parser = argparse.ArgumentParser(description="TensorFlow implementation of Policy Gradient")
parser.add_argument("--n_eps", type=int, default=1000, help="Number of episodes for training.")
parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor (gamma).")
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
action_size = env.action_space.shape[0]


def weight_variable(shape, name):
    """ Return a tensorflow.Variable for dense layer weights. """
    init_val = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_val, name=name)


def bias_variable(shape, name):
    """ Return a tensorflow.Variable for bias. """
    return tf.Variable(tf.constant(0.1, shape=shape), name=name)


def discount(gamma, rewards):
    """ Return the list of discounted rewards. """
    return [r * (gamma ** t) for t, r in enumerate(rewards)]


def rollout(policy, env, sess, render=False):
    """ Execute the argument policy's rollout in the environment. """
    done = False
    state = env.reset()
    while not done:
        if render:
            env.render()
        state = state.reshape((1, obs_size))
        action = sess.run(policy, feed_dict={observation: state})
        state, reward, done, info = env.step(action.reshape((action_size,)))


# A two-layered policy network with a residual connection and a ReLU nonlinearity.
with tf.name_scope("policy"):
    # Observation of the state.
    with tf.name_scope("observation"):
        observation = tf.placeholder(tf.float32, [None, obs_size])

    # Hidden layer of the policy network.
    with tf.name_scope("hidden_layer"):
        W_1 = weight_variable([obs_size, obs_size], "W_1")
        b_1 = bias_variable([obs_size], "b_1")
        # Input to hidden connections with identity mapping
        a_1 = tf.nn.relu(tf.matmul(observation, W_1 + tf.eye(obs_size)) + b_1)

    # Output layer of the policy network.
    with tf.name_scope("output_layer"):
        W_2 = weight_variable([obs_size, action_size], "W_2")
        b_2 = bias_variable([action_size], "b_2")
        policy = tf.matmul(a_1, W_2) + b_2

# Given the output of the policy network, compute the policy gradient.
with tf.name_scope("policy_gradient"):
    reward = tf.placeholder(tf.float32, [None, 1])
    policy_gradient = tf.reduce_mean(tf.log(policy) * reward)
    train_op = tf.train.AdamOptimizer().minimize(-policy_gradient)

with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter("./", graph=sess.graph)
    sess.run(tf.global_variables_initializer())
    trajectory = []
    for ep in range(args.n_eps):
        s = env.reset()
        while True:
            a = sess.run(policy, feed_dict={observation: s.reshape([1, obs_size])})
            s_1, r, done, info = env.step(a.reshape([action_size]))
            trajectory.append((s, a, r))
            if done:
                states = np.stack([t[0].reshape([obs_size]) for t in trajectory], axis=0)
                discounted = discount(args.gamma, [t[2] for t in trajectory])
                dc_rewards = np.stack(discounted, axis=0).reshape([len(trajectory), 1])
                feed_dict = {observation: states, reward: dc_rewards}
                with tf.name_scope("return"):
                    g_t = tf.constant(np.sum(dc_rewards))
                tf.summary.scalar("return", g_t)

                merged = tf.summary.merge_all()
                summary_str, ret_val, _ = sess.run([merged, g_t, train_op], feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, ep)
                if ep % (args.n_eps / 10) == 0:
                    print("Episode %d: return = %f" % (ep, ret_val))
                break
            s = s_1

    # Test rollout.
    rollout(policy, env, sess, render=True)
