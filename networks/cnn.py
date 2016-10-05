import os
import tensorflow as tf

from .layers import *
from .network import Network

class CNN(Network):
  def __init__(self, sess,
               data_format,
               history_length,
               observation_dims,
               output_size, 
               trainable=True,
               hidden_activation_fn=tf.nn.relu,
               output_activation_fn=None,
               weights_initializer=initializers.xavier_initializer(),
               biases_initializer=tf.constant_initializer(0.1),
               value_hidden_sizes=[512],
               advantage_hidden_sizes=[512],
               network_output_type='dueling',
               network_header_type='nips',
               name='CNN'):
    super(CNN, self).__init__(sess, name)

    if data_format == 'NHWC':
      self.inputs = tf.placeholder('float32',
          [None] + observation_dims + [history_length], name='inputs')
    elif data_format == 'NCHW':
      self.inputs = tf.placeholder('float32',
          [None, history_length] + observation_dims, name='inputs')
    else:
      raise ValueError("unknown data_format : %s" % data_format)
    print self.inputs.get_shape()
    self.var = {}
    self.l0 = tf.div(self.inputs, 255.)
    with tf.variable_scope(name):
      if network_header_type.lower() == 'nature':
        self.l1, self.var['l1_w'], self.var['l1_b'] = conv2d(self.l0,
            64, [7, 7], [3, 3], weights_initializer, biases_initializer,
            hidden_activation_fn, data_format, name='l1_conv')
        print self.var['l1_w'].get_shape()
        print self.var['l1_b'].get_shape()
        print self.l1.get_shape()
        self.l2, self.var['l2_w'], self.var['l2_b'] = conv2d(self.l1,
            64, [3, 3], [1, 1], weights_initializer, biases_initializer,
            hidden_activation_fn, data_format, name='l2_conv', padding='SAME')
        print self.var['l2_w'].get_shape()
        print self.var['l2_b'].get_shape()
        print self.l2.get_shape()
        self.l3, self.var['l3_w'], self.var['l3_b'] = conv2d(self.l2,
            64, [3, 3], [1, 1], weights_initializer, biases_initializer,
            hidden_activation_fn, data_format, name='l3_conv', padding='SAME')
        print self.var['l3_w'].get_shape()
        print self.var['l3_b'].get_shape()
        self.l33 = tf.add(self.l3, self.l1)
        self.l4, self.var['l4_w'], self.var['l4_b'] = \
            linear(self.l3, 512, weights_initializer, biases_initializer,
            hidden_activation_fn, data_format, name='l4_conv')
        print self.var['l4_w'].get_shape()
        print self.var['l4_b'].get_shape()
        layer = self.l4
      elif network_header_type.lower() == 'nips':
        self.l1, self.var['l1_w'], self.var['l1_b'] = conv2d(self.l0,
            16, [8, 8], [4, 4], weights_initializer, biases_initializer,
            hidden_activation_fn, data_format, name='l1_conv')
        self.l2, self.var['l2_w'], self.var['l2_b'] = conv2d(self.l1,
            32, [4, 4], [2, 2], weights_initializer, biases_initializer,
            hidden_activation_fn, data_format, name='l2_conv')
        self.l3, self.var['l3_w'], self.var['l3_b'] = \
            linear(self.l2, 256, weights_initializer, biases_initializer,
            hidden_activation_fn, data_format, name='l3_conv')
        layer = self.l3
      else:
        raise ValueError('Wrong DQN type: %s' % network_header_type)

      self.build_output_ops(layer, network_output_type,
          value_hidden_sizes, advantage_hidden_sizes, output_size,
          weights_initializer, biases_initializer, hidden_activation_fn,
          output_activation_fn, trainable)
