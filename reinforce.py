# TensorFlow implementation of Policy Gradient (REINFORCE)

import argparse
import gym
import roboschool
import numpy as np
import tensorflow as tf

def swish(x, beta):
    return x * tf.sigmoid(beta * x)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

class Policy(object):
    def __init__(self, input_size, hidden_size, output_size):
        # Hidden layer
        self.W_1 = weight_variable((input_size, hidden_size))
        self.b_1 = bias_variable(hidden_size)
        
        # Beta variable for swish activation function.
        beta_initial = tf.truncated_normal(hidden_size, stddev=0.1)
        self.beta = tf.Variable(beta_initial)

        # Output layer
        self.W_2 = weight_variable((hidden_size, output_size))
        self.b_2 = bias_variable(output_size)

    def __call__(self, s):
        return self._forward(s)

    def _forward(self, s):
        hidden = swish(tf.matmul(s, self.W_1) + self.b_1, self.beta)
        return tf.matmul(hidden, self.W_2) + self.b_2

def main(args):
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TensorFlow implementation of REINFORCE')
    parser.add_argument('--env', str, '')
    args = parser.parse_args()

    main(args)